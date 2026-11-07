"""补 STE 臂缺的三个 seed（0/6/7），split_seed 固定 0，与覆盖臂配对。

已有 fixed-split STE：s1 1793467903、s2 1793468803、s3 1793469716、s4 1793470599、s5 1793471551。
本脚本补 s0/s6/s7，使 STE 与 none@N=1 在 split_seed=0 上各有 8 个 seed，可做 run 层配对。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PY = r"C:/Users/24399/.conda/envs/e2e/python.exe"
ROOT = Path(__file__).resolve().parents[1]
BASE = ["train.py", "--head", "relu_qat", "--ema-mode", "none", "--ema-decay", "0.99",
        "--snap-every", "1", "--epochs", "20", "--batch-size", "64", "--lr", "1e-4",
        "--amp", "--grad-clip", "1.0", "--raw", "data/utkface_raw224.npy",
        "--quant-mode", "ste", "--split-seed", "0"]
SEEDS = [0, 6, 7]


def main() -> None:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    rcs = []
    for s in SEEDS:
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
