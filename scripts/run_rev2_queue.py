"""第二批审稿意见的补跑队列（P1 N=∞ / P2 补种子 / P3 γ 扫描 / P4 固定划分）。

用法: python scripts/run_rev2_queue.py A|B
已完成的配置会被跳过（按 results/*/result.json 的 args 判重），因此可重复启动。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
BASE = [PY, "-u", "train.py", "--head", "relu_qat", "--epochs", "20", "--batch-size", "64",
        "--lr", "1e-4", "--amp", "--grad-clip", "1.0", "--raw", "data/utkface_raw224.npy",
        "--workers", "6"]
INF = 0  # snap-every<=0 表示训练期从不投影（N=∞ 臂）

RUNS = {
    "A": [
        ("P1-none-inf", dict(snap=INF, mode="none", seed=0)),
        ("P4-split0-init1", dict(snap=20, mode="after_snap", seed=1, split=0)),
        ("P2-n5-none-s1", dict(snap=5, mode="none", seed=1)),
        ("P2-n5-before-s1", dict(snap=5, mode="before_snap", seed=1)),
        ("P2-n5-after-s1", dict(snap=5, mode="after_snap", seed=1)),
        ("P3-g09-before", dict(snap=20, mode="before_snap", seed=0, decay=0.9)),
        ("P3-g09-after", dict(snap=20, mode="after_snap", seed=0, decay=0.9)),
    ],
    "B": [
        ("P1-after-inf", dict(snap=INF, mode="after_snap", seed=0)),
        ("P4-split0-init2", dict(snap=20, mode="after_snap", seed=2, split=0)),
        ("P4-split0-init3", dict(snap=20, mode="after_snap", seed=3, split=0)),
        ("P2-n5-none-s2", dict(snap=5, mode="none", seed=2)),
        ("P2-n5-before-s2", dict(snap=5, mode="before_snap", seed=2)),
        ("P2-n5-after-s2", dict(snap=5, mode="after_snap", seed=2)),
        ("P3-g0999-before", dict(snap=20, mode="before_snap", seed=0, decay=0.999)),
        ("P3-g0999-after", dict(snap=20, mode="after_snap", seed=0, decay=0.999)),
    ],
}


def done_set() -> set[tuple]:
    out = set()
    for rj in Path("results").glob("utk-relu_qat-*/result.json"):
        try:
            a = json.loads(rj.read_text(encoding="utf-8")).get("args", {})
        except Exception:
            continue
        out.add((int(a.get("snap_every", 1)), a.get("ema_mode"), int(a.get("seed", 0)),
                 round(float(a.get("ema_decay", 0.99)), 4),
                 int(a.get("split_seed", -1)) if a.get("split_seed") is not None else -1))
    return out


def main() -> None:
    q = sys.argv[1] if len(sys.argv) > 1 else "A"
    runs = RUNS[q]
    already = done_set()
    for tag, spec in runs:
        key = (spec["snap"], spec["mode"], spec["seed"], round(spec.get("decay", 0.99), 4),
               spec.get("split", -1))
        if key in already:
            print(f"[{q}] SKIP {tag} (already present)", flush=True)
            continue
        cmd = BASE + ["--ema-mode", spec["mode"], "--snap-every", str(spec["snap"]),
                      "--seed", str(spec["seed"])]
        if "decay" in spec:
            cmd += ["--ema-decay", str(spec["decay"])]
        if "split" in spec:
            cmd += ["--split-seed", str(spec["split"])]
        print(f"[{q}] >>> {tag} :: {' '.join(cmd[2:])}", flush=True)
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        tail = (p.stdout or "").strip().splitlines()[-4:]
        print(f"[{q}] <<< {tag} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
        print("\n".join(tail) or (p.stderr or "")[-800:], flush=True)
    print(f"[{q}] QUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
