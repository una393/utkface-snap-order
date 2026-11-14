"""STE 臂的正式队列：seed 0/1/2，超参与现有 N=1 的 none 臂逐项一致。

    --head relu_qat --ema-mode none --ema-decay 0.99 --snap-every 1 --epochs 20
    --batch-size 64 --lr 1e-4 --amp --grad-clip 1.0 --raw data/utkface_raw224.npy
    --quant-mode ste --seed S            （--split-seed 默认 = seed）

run 目录名：results/utk-relu_qat-none-ste-s{S}-{ts}。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PY = r"C:/Users/USER/.conda/envs/e2e/python.exe"
ROOT = Path(__file__).resolve().parents[1]
BASE = ["train.py", "--head", "relu_qat", "--ema-mode", "none", "--ema-decay", "0.99",
        "--snap-every", "1", "--epochs", "20", "--batch-size", "64", "--lr", "1e-4",
        "--amp", "--grad-clip", "1.0", "--raw", "data/utkface_raw224.npy",
        "--quant-mode", "ste"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    rcs = []
    for s in [int(t) for t in a.seeds.split(",") if t.strip()]:
        cmd = [PY] + BASE + ["--seed", str(s)]
        print("RUN " + " ".join(cmd), flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        rcs.append((s, rc))
        print(f"DONE seed={s} rc={rc} wall={time.time() - t0:.0f}s", flush=True)
    print("SUMMARY " + ", ".join(f"s{s}:rc{rc}" for s, rc in rcs), flush=True)
    sys.exit(0 if all(rc == 0 for _, rc in rcs) else 1)


if __name__ == "__main__":
    main()
