"""STE + 影子平均的正式队列：seed 0/1/2，补 paper §7 里「该组合尚未定义」的空缺。

与 scripts/run_ste_queue.py 同配置（同超参、同 STE 前向伪量化），差别只有两处：
--ema-mode after_snap（要维护影子）与 --ema-source forward（影子平均的是前向用的
量化权重 wq，不是浮点原始参数）。其余一行不动。

    --dataset utkface --raw data/utkface_raw224.npy --arch net --head relu_qat
    --ema-source forward --quant-mode ste --ema-mode after_snap --ema-decay 0.99
    --snap-every 1 --epochs 20 --batch-size 64 --lr 1e-4 --amp --grad-clip 1.0
    --workers 6 --seed S                 （--split-seed 默认 = seed）

run 目录名：results/utk-relu_qat-after_snap-ste-s{S}-{ts}。
注意 ste 下 snap 一次都不被调用，history.csv 的 snap_err/frozen_frac 因此是 nan；
影子由 --ema-mode after_snap 照旧每步平均（值为 nan 的那两列只是没被调用，不是错）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PY = r"C:/Users/24399/.conda/envs/e2e/python.exe"
ROOT = Path(__file__).resolve().parents[1]
BASE = ["train.py", "--dataset", "utkface", "--raw", "data/utkface_raw224.npy",
        "--arch", "net", "--head", "relu_qat",
        "--ema-source", "forward", "--quant-mode", "ste", "--ema-mode", "after_snap",
        "--ema-decay", "0.99", "--snap-every", "1", "--epochs", "20",
        "--batch-size", "64", "--lr", "1e-4", "--amp", "--grad-clip", "1.0",
        "--workers", "6"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行、不建 run 目录")
    a = ap.parse_args()
    seeds = [int(t) for t in a.seeds.split(",") if t.strip()]
    cmds = [[PY] + BASE + ["--seed", str(s)] for s in seeds]
    if a.dry_run:
        for i, cmd in enumerate(cmds, 1):
            print(f"[dry-run {i}/{len(cmds)}] " + " ".join(cmd), flush=True)
        return
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    rcs = []
    for seed, cmd in zip(seeds, cmds):
        print("RUN " + " ".join(cmd), flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        rcs.append((seed, rc))
        print(f"DONE seed={seed} rc={rc} wall={time.time() - t0:.0f}s", flush=True)
        if rc != 0:   # 失败即停：后面两个 seed 不跑
            print(f"STOP seed={seed} rc={rc}", flush=True)
            sys.exit(rc)
    print("SUMMARY " + ", ".join(f"s{s}:rc{rc}" for s, rc in rcs), flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
