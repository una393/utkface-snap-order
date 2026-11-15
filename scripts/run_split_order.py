"""Cross-split replication of the ordering contrast.

The ordering claim (ema_before vs ema_after at interval 20) was measured on one
split with eight seeds. Splits 1 and 2 already carry ema_after runs from the 3x3
design, but only one ema_before run each, so the paired contrast exists on those
splits for a single seed. This queue completes the 3 x 3 cells
(split_seed, seed) = (1,2) (1,3) (2,1) (2,3) with ema_before at interval 20, the
same configuration as the main before_snap arm, so the contrast can be read per
split and pooled.

A cell already present in results/ is skipped (matched on result.json args, the
directory name does not carry split_seed), so the queue can be restarted. A
training subprocess returncode != 0 aborts the queue instead of being counted as
a success.

Usage: python scripts/run_split_order.py [--dry-run]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
BASE = [PY, "-u", "train.py", "--raw", "data/utkface_raw224.npy",
        "--head", "relu_qat", "--ema-mode", "before_snap", "--ema-decay", "0.99",
        "--epochs", "20", "--batch-size", "64", "--lr", "1e-4", "--amp",
        "--grad-clip", "1.0", "--workers", "6", "--snap-every", "20",
        "--quant-mode", "overwrite"]
CELLS = [(1, 2), (1, 3), (2, 1), (2, 3)]


def done_set() -> set[tuple[int, int]]:
    out = set()
    for rj in Path("results").glob("utk-*/result.json"):
        try:
            a = json.loads(rj.read_text(encoding="utf-8")).get("args", {})
        except Exception:
            continue
        if a.get("head") != "relu_qat" or a.get("ema_mode") != "before_snap":
            continue
        if a.get("epochs") != 20 or int(a.get("snap_every") or 0) != 20:
            continue
        if a.get("quant_mode", "overwrite") != "overwrite":
            continue
        if round(float(a.get("ema_decay", 0.99)), 4) != 0.99:
            continue
        seed = int(a.get("seed", 0))
        sp = a.get("split_seed", -1)
        sp = seed if sp is None or int(sp) < 0 else int(sp)
        out.add((int(sp), seed))
    return out


def main() -> None:
    dry = "--dry-run" in sys.argv
    already = done_set()
    print("present cells:", sorted(already), flush=True)
    for sp, seed in CELLS:
        if (sp, seed) in already:
            print("[X] SKIP split=%d seed=%d (already present)" % (sp, seed), flush=True)
            continue
        cmd = BASE + ["--split-seed", str(sp), "--seed", str(seed)]
        print("[X] >>> split=%d seed=%d" % (sp, seed), flush=True)
        if dry:
            continue
        t0 = time.time()
        rc = subprocess.call(cmd)
        dt = time.time() - t0
        print("[X] split=%d seed=%d rc=%d %.0f s" % (sp, seed, rc, dt), flush=True)
        if rc != 0:
            raise SystemExit("training failed for split=%d seed=%d" % (sp, seed))
    print("[X] queue done", flush=True)


if __name__ == "__main__":
    main()
