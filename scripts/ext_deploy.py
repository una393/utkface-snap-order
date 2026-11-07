"""架构感知的部署评估器：结果写独立 csv，绝不进 quant_runs/deploy_table_v2.csv。

deploy_table.py 只服务 UTKFace × DrivingNet 那一张表（它对 args.arch 有硬守卫，也是
论文 2 的既有产物来源）。这条链路复用同一套 qdq_ 口径，把架构轴打开：

* arch=net     —— DrivingNet(head=args.head, grid=(14,14), seq=1)，直接前向；
* arch=resnet  —— ResNetReg(pretrained=True)，训练期 BN 还在权重里，量化前必须先
                  fold_bn_（见 quant/bnfold.py），否则部署权重里留 BatchNorm。

与 deploy_table.py 共用的口径（必须是同一条实现，不许分叉）：
1) best.pth 的取值键：ema_mode == "none" 取 model_state_dict，其余一律取 ema_state_dict
   —— 与 deploy_table.py:148 的 `key = "ema_state_dict" if mode != "none" else "model_state_dict"`
   逐字一致；所需键缺失时直接 raise 并指名 run 与键，**不回退**到另一个键（none 的 run 里
   ema_state_dict 存在但从未更新，回退等于拿初始化权重当部署权重）；
2) 默认扫目录时跳过名字含 "-ste-" 的目录（同 deploy_table.py:89）：STE 臂是同 (none, N=1)
   的另一个算法，混进部署表就是重复行；本过滤器对显式 --runs 同样生效。

* dataset=koniq —— records 从 --index 的索引 csv 读，验证划分用 data.koniq.split_random，
  与 UTKFace 完全同构（memmap、qdq_、取值键都不变）。

--subset 决定评估用哪些验证图（只影响选谁，不影响量化口径）：
* first400 —— 验证划分前 n 张（改造前的行为，n 默认 400）；
* heldout   —— 验证划分减去「训练期用于选轮的子集」。该子集按 train.py:77-79 同式重建：
               random.Random(split_seed).sample(val, --selection-windows)。len(val) <=
               selection_windows 时 heldout 为空（训练根本没做子采样），直接 raise。

与 deploy_table.py 的有意差异（仅这两处）：
1) 载入一律 strict=True —— 跨架构的键不能靠 strict=False 蒙混；
2) 每个 bit 都从干净模型的深拷贝出发 —— 8 bit 的量化结果不会污染 6 bit。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from data import koniq
from data.utkface import scan, split_random
from models.net import DrivingNet
from models.resnet_reg import ResNetReg
from quant.bnfold import fold_bn_
from scripts.deploy_table import MEAN, STD, qdq_, split_seed_of
from train import read_index_csv   # records 口径只有一份：与 train.py 逐位同一条实现


UTK_RAW = "data/utkface_raw224.npy"
KONIQ_RAW = "data/koniq_raw224.npy"


def load_records(dataset: str, data: str, index: str) -> tuple[list[dict], object]:
    """(records, split_random 实现)。两边的 split_random 语义相同，只是模块不同。"""
    if dataset == "koniq":
        return read_index_csv(index), koniq.split_random
    return scan(data), split_random


def pick_records(val: list[dict], subset: str, n: int, selection_windows: int,
                 split_seed: int) -> list[dict]:
    """验证划分 → 本次要评估的那批图（顺序保持划分原序）。"""
    if subset == "first400":
        return val[:n]
    if len(val) <= selection_windows:
        raise ValueError(
            "subset=heldout 需要 len(val) > --selection-windows；当前 val=%d、selection-windows=%d，"
            "heldout 会为空（该划分下训练期没有做子采样，全部验证图都参与过选轮）"
            % (len(val), selection_windows))
    sel_ids = {id(r) for r in random.Random(split_seed).sample(val, selection_windows)}
    return [r for r in val if id(r) not in sel_ids]


def run_dirs(pattern: str, include_ste: bool = False) -> list[Path]:
    """扫描口径：默认跳过 STE 臂（同 deploy_table.py:89），只留带 result.json 的 run。

    include_ste=True 时不再过滤 -ste-：STE×EMA 臂本身就是 ste run，要单独出表。
    """
    return [d for d in sorted(Path(".").glob(pattern))
            if (include_ste or "-ste-" not in d.name) and (d / "result.json").exists()]


def build_net(arch: str, args: dict, dev: torch.device) -> torch.nn.Module:
    if arch == "net":
        net = DrivingNet(head=args["head"], grid=(14, 14), seq=1)
    elif arch == "resnet":
        net = ResNetReg(pretrained=True)
    else:
        raise ValueError("未知 arch %r（只有 net / resnet）" % arch)
    return net.to(dev)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/utk-relu_qat-*", help="run 目录的 glob")
    p.add_argument("--out", default="quant_runs/ext_deploy.csv")
    p.add_argument("--bits", default="32,8,6,4")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--raw", default="", help="空 = 按 --dataset 取自带缓存")
    p.add_argument("--data", default="D:/data/utkface/UTKFace", help="仅 --dataset utkface")
    p.add_argument("--dataset", choices=["utkface", "koniq"], default="utkface")
    p.add_argument("--index", default="data/koniq_index.csv", help="koniq 索引 csv")
    p.add_argument("--selection-windows", type=int, default=2000,
                   help="训练期用于选轮的验证图数量（重建 heldout 用）")
    p.add_argument("--subset", choices=["first400", "heldout"], default="first400")
    p.add_argument("--include-ste", action="store_true",
                   help="不过滤 -ste- 目录（STE×EMA 臂专用）")
    a = p.parse_args()
    if not a.raw:
        a.raw = KONIQ_RAW if a.dataset == "koniq" else UTK_RAW
    bits = [int(b) for b in a.bits.split(",")]
    if a.subset == "heldout":
        print("subset=heldout：n 由反选集决定，--n 只约束 first400", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    recs, split_fn = load_records(a.dataset, a.data, a.index)
    arr = np.load(a.raw, mmap_mode="r")
    splits: dict[int, list] = {}
    keys = ["run", "dataset", "subset", "arch", "head", "mode", "N", "split_seed", "seed", "n",
            "mae_p32", "mae_8", "mae_6", "mae_4", "source"]

    rows = []
    for d in run_dirs(a.runs, a.include_ste):
        meta = json.loads((d / "result.json").read_text(encoding="utf-8"))
        args = meta["args"]
        arch = args.get("arch", "net")
        seed = args.get("seed", 0)
        mode = args["ema_mode"]
        sp = split_seed_of(args, seed, d.name)
        if sp not in splits:
            splits[sp] = pick_records(split_fn(recs, 0.15, sp)[1], a.subset, a.n,
                                      a.selection_windows, sp)
        sel = splits[sp]
        xs, ys = [], []
        for r in sel:
            x = np.asarray(arr[int(r["row"])], dtype="float32") / 255.0
            x = (x - MEAN) / STD
            xs.append(torch.from_numpy(x.transpose(2, 0, 1)[None, None].copy()))
            ys.append(float(r["target"]))
        ys_t = torch.tensor(ys).to(dev)

        ck = torch.load(d / "best.pth", map_location="cpu")
        key = "ema_state_dict" if mode != "none" else "model_state_dict"
        if key not in ck:
            raise KeyError("%s: best.pth 缺 %r（ema_mode=%r 决定的取值键；不回退到另一个键）"
                           % (d.name, key, mode))
        clean = build_net(arch, args, dev)
        clean.load_state_dict(ck[key], strict=True)
        clean.eval()
        if arch == "resnet":
            fold_bn_(clean)

        def forward_all(model: torch.nn.Module) -> torch.Tensor:
            outs = []
            with torch.no_grad():
                for i in range(0, len(xs), 64):
                    outs.append(model(torch.cat(xs[i:i + 64], 0).to(dev)).flatten())
            return torch.cat(outs)

        row = {"run": d.name, "dataset": a.dataset, "subset": a.subset,
               "arch": arch, "head": args["head"],
               "mode": mode, "N": args.get("snap_every", 1),
               "split_seed": sp, "seed": seed, "n": len(sel), "source": "computed"}
        for b in bits:
            net = copy.deepcopy(clean)
            if b < 32:
                qdq_(net, b)
            y = forward_all(net)
            row["mae_p32" if b >= 32 else "mae_%d" % b] = round(float((y - ys_t).abs().mean()), 6)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print("wrote %s (%d rows)" % (out, len(rows)), flush=True)


if __name__ == "__main__":
    main()
