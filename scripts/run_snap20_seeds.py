"""freq=20 的 seed 复核：把「4 bit 下较少吸附更稳」从单 seed 观察变成 n=3。"""
from __future__ import annotations

import subprocess
import sys

PY = sys.executable
for seed in (1, 2):
    for mode in ("before_snap", "after_snap"):
        cmd = [PY, "-u", "train.py", "--head", "relu_qat", "--ema-mode", mode,
               "--snap-every", "20", "--epochs", "20", "--batch-size", "64", "--lr", "1e-4",
               "--seed", str(seed), "--amp", "--grad-clip", "1.0",
               "--raw", "data/utkface_raw224.npy", "--workers", "8"]
        print(f"===== seed={seed} mode={mode}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode:
            sys.exit(f"FAILED rc={p.returncode}: {' '.join(cmd)}\n{(p.stderr or '')[-1500:]}")
        print("\n".join((p.stdout or "").strip().splitlines()[-2:]) or (p.stderr or "")[-400:],
              flush=True)
print("SEEDS DONE", flush=True)
