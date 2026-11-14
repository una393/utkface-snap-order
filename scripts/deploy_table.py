"""部署量化后的任务侧 MAE：3（吸附频率）× 3（更新时机）的最终表。

把部署权重按 bits 量化再反量化，前向，与真实年龄标签比 MAE。FP32 一行作为上界对照。

⚠️ 验证集必须按**该 run 自己的 seed** 重建：train.py 用 split_random(recs, 0.15, seed)，
不同 seed 的验证集不同。用固定 seed 0 去评 s1/s2 会踩到自己训练过的帧（实测把 MAE
从 0.0579 压到 0.0389），这是数据泄漏，不是模型好。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn

from data.utkface import scan, split_random
from models.net import DrivingNet

MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")


def qdq_(net: nn.Module, bits: int) -> None:
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for m in net.modules():
            if not isinstance(m, (nn.Linear, nn.Conv2d)):
                continue
            w = m.weight.data
            flat = w.reshape(w.shape[0], -1)
            scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
            q = (flat / scale).round().clamp(-qmax - 1, qmax)
            m.weight.data = (q * scale).reshape(w.shape)


SPLIT_SEED_SINCE = 1793111850   # 最早记录 split_seed 的 run；再早的 run 无此选项，划分硬编码成 seed


def split_seed_of(args: dict, seed: int, run: str) -> int:
    """-1/None 是 train.py 的默认语义（= seed）。键缺失只在 --split-seed 之前的 run 上合法：
    result.json 存的是 vars(args) 全量（train.py:177），那时的划分恒等于 seed。"""
    if "split_seed" in args:
        sp = args["split_seed"]
        return seed if sp is None or int(sp) < 0 else int(sp)
    if int(run.rsplit("-", 1)[-1]) >= SPLIT_SEED_SINCE:
        raise KeyError("%s: result.json 的 args 缺 split_seed（--split-seed 之后产生的 run 必须记录），"
                       "无法判定验证划分" % run)
    return seed


def decay_of(args: dict, run: str) -> float:
    if "ema_decay" not in args:
        raise KeyError("%s: result.json 的 args 缺 ema_decay，无法判定是否属于主配置 γ=0.99" % run)
    return args["ema_decay"]


def arch_of(args: dict, run: str) -> str:
    """跨架构守卫：本表只收 DrivingNet（net）的 run，别的架构混进来就是污染。

    args 缺 arch 键的 run 早于 --arch（train.py 10-31 才加），按 net 处理。
    """
    arch = args.get("arch", "net")
    if arch != "net":
        raise ValueError("%s: args.arch=%r，本表只收 DrivingNet（net）的 run；"
                         "跨架构结果请用 scripts/ext_deploy.py" % (run, arch))
    return arch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/utkface_raw224.npy")
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--bits", default="8,6,4")
    p.add_argument("--out", default="quant_runs/deploy_table.csv")
    p.add_argument("--only-freq", default="", help="只跑这些吸附频率，逗号分隔")
    p.add_argument("--only-seed", default="", help="只跑这些 seed，逗号分隔")
    p.add_argument("--dump", default="", help="落盘逐图预测与标签的目录（配对统计用）")
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    recs = scan(a.data)
    arr = np.load(a.raw, mmap_mode="r")
    splits: dict[int, list] = {}
    dump_dir = Path(a.dump) if a.dump else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sorted(Path("results").glob("utk-relu_qat-*")):
        if "-ste-" in d.name:      # STE 臂是同 (none, N=1) 的另一个算法，混进来会变成重复行
            continue
        if "-resnet-" in d.name:   # 跨架构 run（ResNet-18 轴）不是 DrivingNet，见 arch_of()
            continue
        rj = d / "result.json"
        if not rj.exists():
            continue
        meta = json.loads(rj.read_text(encoding="utf-8"))
        args = meta["args"]
        arch_of(args, d.name)
        if args["head"] != "relu_qat":
            continue
        freq = args.get("snap_every", 1)
        mode = args["ema_mode"]
        seed = args.get("seed", 0)
        if a.only_freq and str(freq) not in a.only_freq.split(","):
            continue
        if a.only_seed and str(seed) not in a.only_seed.split(","):
            continue

        sp = split_seed_of(args, seed, d.name)
        decay = decay_of(args, d.name)
        need = ["fp32"] + [f"{b}bit" for b in a.bits.split(",")]
        if dump_dir and all((dump_dir / f"{d.name}__{k}.npz").exists() for k in need):
            row = {"snap_every": freq, "mode": mode, "seed": seed, "run": d.name,
                   "ema_decay": decay, "split_seed": sp,
                   "val_mae_train": round(float(meta["val_mae_live"]), 4)}
            for k in need:
                z = np.load(dump_dir / f"{d.name}__{k}.npz")
                row["mae_fp32" if k == "fp32" else f"mae_{k[:-3]}"] = \
                    round(float(np.abs(z["pred"] - z["target"]).mean()), 4)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            continue
        if sp not in splits:
            splits[sp] = split_random(recs, 0.15, sp)[1][: a.n]
        sel = splits[sp]
        rows_sel = np.array([int(r["row"]) for r in sel])
        xs, ys = [], []
        for r in sel:
            x = np.asarray(arr[int(r["row"])], dtype="float32") / 255.0
            x = (x - MEAN) / STD
            xs.append(torch.from_numpy(x.transpose(2, 0, 1)[None, None].copy()))
            ys.append(float(r["target"]))
        ys_t = torch.tensor(ys).to(dev)

        ck = torch.load(d / "best.pth", map_location="cpu")
        key = "ema_state_dict" if mode != "none" else "model_state_dict"
        net = DrivingNet(head="relu_qat", grid=(14, 14), seq=1).to(dev)
        net.load_state_dict(ck[key], strict=False)
        net.eval()

        row = {"snap_every": freq, "mode": mode, "seed": seed, "run": d.name,
               "ema_decay": decay, "split_seed": sp,
               "val_mae_train": round(float(meta["val_mae_live"]), 4)}
        def forward_all(model):
            outs = []
            with torch.no_grad():
                for i in range(0, len(xs), 64):
                    outs.append(model(torch.cat(xs[i:i + 64], 0).to(dev)).flatten())
            return torch.cat(outs)

        with torch.no_grad():
            base = {k: v.detach().clone() for k, v in net.state_dict().items()}
            y = forward_all(net)
            row["mae_fp32"] = round(float((y - ys_t).abs().mean()), 4)
            if dump_dir:
                np.savez_compressed(dump_dir / f"{d.name}__fp32.npz", pred=y.detach().cpu().numpy(),
                                    target=ys_t.detach().cpu().numpy(), rows=rows_sel)
            for bits in [int(b) for b in a.bits.split(",")]:
                net.load_state_dict(base, strict=False)
                qdq_(net, bits)
                yq = forward_all(net)
                row[f"mae_{bits}"] = round(float((yq - ys_t).abs().mean()), 4)
                if dump_dir:
                    np.savez_compressed(dump_dir / f"{d.name}__{bits}bit.npz", pred=yq.detach().cpu().numpy(),
                                        target=ys_t.detach().cpu().numpy(), rows=rows_sel)
        net.load_state_dict(base, strict=False)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["snap_every", "mode", "seed", "ema_decay", "split_seed", "run",
            "val_mae_train", "mae_fp32"] + [f"mae_{b}" for b in a.bits.split(",")]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
