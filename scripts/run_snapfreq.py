"""吸附频率 × EMP 更新时机 的扫频训练（篇 2 的第二个自变量）。

已有 snap_every=1 的三组（none / before_snap / after_snap），这里补 5 和 20。
判据先定死：只比同 snap_every 下三组的部署权重量化余弦，不与 snap_every=1 比绝对 MAE。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
CFGS = [(n, m) for n in (5, 20) for m in ("none", "before_snap", "after_snap")]

for n, m in CFGS:
    cmd = [PY, "-u", "train.py", "--head", "relu_qat", "--ema-mode", m, "--snap-every", str(n),
           "--epochs", "20", "--batch-size", "64", "--lr", "1e-4", "--seed", "0",
           "--amp", "--grad-clip", "1.0", "--raw", "data/utkface_raw224.npy", "--workers", "8"]
    print(f"===== snap_every={n} mode={m}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode:
        sys.exit(f"FAILED rc={p.returncode}: {' '.join(cmd)}\n{(p.stderr or '')[-1500:]}")
    tail = (p.stdout or "").strip().splitlines()[-3:]
    print("\n".join(tail) or (p.stderr or "")[-500:], flush=True)
print("ALL DONE", flush=True)
