"""三控制头的逐图预测导出（同一批 200 张），供 round20 配对检验使用。

划分口径照抄 scripts/deploy_table.py：
  recs = scan(--data)；split_random(recs, 0.15, split_seed)[1][:n] 取验证集前 n 张；
  split_seed 由该 run 自身 result.json 的 args 决定（split_seed_of 的语义：
  args.split_seed 为 -1/None → seed；键缺失且 run 时间戳早于 --split-seed 则恒等于 seed）。
图像来自 --raw 的 uint8 memmap，归一化与 data/transforms.raw_to_tensor 逐位相同。
预测一律 fp32 前向（与 deploy_table.py 一致，不用 autocast）。

输出 quant_runs/head_eval/<run>__<n>.npz：pred / target（float32 一维）、rows（int64 行号）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from data.utkface import scan, split_random
from models.net import DrivingNet

MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")

SPLIT_SEED_SINCE = 1793111850   # 与 scripts/deploy_table.py:44 同一个分界


def split_seed_of(args: dict, seed: int, run: str) -> int:
    """与 scripts/deploy_table.py:47-56 同义：-1/None = seed，键缺失只对老 run 合法。"""
    if "split_seed" in args:
        sp = args["split_seed"]
        return seed if sp is None or int(sp) < 0 else int(sp)
    if int(run.rsplit("-", 1)[-1]) >= SPLIT_SEED_SINCE:
        raise KeyError("%s: result.json 的 args 缺 split_seed，无法判定验证划分" % run)
    return seed


def forward_all(net, xs, dev, bs: int = 64):
    outs = []
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            outs.append(net(torch.cat(xs[i:i + bs], 0).to(dev)).flatten())
    return torch.cat(outs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run 目录（results/utk-...）")
    p.add_argument("--n", type=int, default=200, help="验证集前 n 张")
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--raw", default="data/utkface_raw224.npy")
    p.add_argument("--out", default="quant_runs/head_eval")
    p.add_argument("--split-seed", type=int, default=-1,
                   help="覆盖：强制用这一个划分种子（默认按每个 run 自己的 result.json）")
    p.add_argument("--test-set", default="npu_quant/test_set.json",
                   help="只为报告重叠张数（npv_quant 的 200 张与本轮 200 张是两套图）")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    recs = scan(a.data)
    arr = np.load(a.raw, mmap_mode="r")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_ref, rows_ref, summaries, tag = None, None, [], None
    for run in a.runs:
        d = Path(run)
        meta = json.loads((d / "result.json").read_text(encoding="utf-8"))
        args = meta["args"]
        head, mode = args["head"], args["ema_mode"]
        seed = int(args.get("seed", 0))
        sp = a.split_seed if a.split_seed >= 0 else split_seed_of(args, seed, d.name)

        sel = split_random(recs, 0.15, sp)[1][: a.n]
        rows = np.array([int(r["row"]) for r in sel], dtype="int64")
        ys = np.array([float(r["target"]) for r in sel], dtype="float32")
        xs = []
        for r in sel:
            x = np.asarray(arr[int(r["row"])], dtype="float32") / 255.0
            x = (x - MEAN) / STD
            xs.append(torch.from_numpy(x.transpose(2, 0, 1)[None, None].copy()))

        net = DrivingNet(head=head, grid=(14, 14), seq=1, pretrained=False).to(dev)
        ck = torch.load(d / "best.pth", map_location="cpu")
        key = "ema_state_dict" if mode != "none" else "model_state_dict"
        missing, unexpected = net.load_state_dict(ck[key], strict=False)
        if missing:
            raise RuntimeError("%s: 缺 %d 个权重键，例 %s" % (d.name, len(missing), missing[:3]))
        if unexpected:
            raise RuntimeError("%s: 多出 %d 个权重键，例 %s" % (d.name, len(unexpected), unexpected[:3]))
        net.eval()
        pred = forward_all(net, xs, dev).detach().cpu().numpy().astype("float32")
        del net
        torch.cuda.empty_cache()

        assert pred.shape == ys.shape == rows.shape == (a.n,), (pred.shape, ys.shape)
        mae = float(np.abs(pred - ys).mean())
        np.savez_compressed(out_dir / ("%s__%d.npz" % (d.name, a.n)),
                            pred=pred, target=ys, rows=rows)
        summaries.append({"run": d.name, "head": head, "ema_mode": mode, "seed": seed,
                          "split_seed": sp, "val_mae_live": round(float(meta["val_mae_live"]), 6),
                          "mae_%d" % a.n: round(mae, 6)})
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)

        if targets_ref is None:
            targets_ref, rows_ref, tag = ys, rows, d.name
        else:
            assert np.array_equal(ys, targets_ref), \
                "target 不一致：%s vs %s" % (d.name, tag)
            assert np.array_equal(rows, rows_ref), "rows 不一致：%s vs %s" % (d.name, tag)

    print("ASSERT OK: %d 个 run 的 target/rows 逐元素相同（n=%d，split_seed 口径一致）"
          % (len(a.runs), a.n), flush=True)
    if tag is not None:
        print("rows[:5] = %s" % rows_ref[:5].tolist(), flush=True)

    ts_path = Path(a.test_set)
    if ts_path.exists():
        ts = json.loads(ts_path.read_text())
        other = set(int(r) for r in ts["rows"][: a.n])
        ov = len(set(int(r) for r in rows_ref) & other)
        print("与 %s 的 %d 张重叠 %d/%d（两套图，报告里必须分开标注）"
              % (ts_path, a.n, ov, a.n), flush=True)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
