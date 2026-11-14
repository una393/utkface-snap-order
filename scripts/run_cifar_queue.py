"""CIFAR-10 的 6 臂 × 8 seed 队列：串行跑，断点续跑（已完成的不重跑）。

臂（--snap-every 0 = 不投影）：
  live@1 / live@5 / live@20 / live@inf   --ema-mode none
  ema_after@20                           --ema-mode after_snap
  ste@1                                  --quant-mode ste（只前向伪量化）
完成记录写在 logs/cifar_runs.jsonl，每行 {arm, seed, dir}。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

PY = sys.executable
STATE = Path("logs/cifar_runs.jsonl")


def arms(snap_every_1: int = 1) -> list[dict]:
    return [
        {"arm": "live@1", "snap": 1, "ema": "none", "quant": "overwrite"},
        {"arm": "live@5", "snap": 5, "ema": "none", "quant": "overwrite"},
        {"arm": "live@20", "snap": 20, "ema": "none", "quant": "overwrite"},
        {"arm": "live@inf", "snap": 0, "ema": "none", "quant": "overwrite"},
        {"arm": "ema_after@20", "snap": 20, "ema": "after_snap", "quant": "overwrite"},
        {"arm": "ste@1", "snap": 1, "ema": "none", "quant": "ste"},
    ]


def done() -> set[tuple[str, int]]:
    if not STATE.exists():
        return set()
    out = set()
    for line in STATE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["arm"], r["seed"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=[], help="只跑这些臂（默认全跑）")
    a = ap.parse_args()

    todo = [x for x in arms() if not a.only or x["arm"] in a.only]
    have = done()
    total = len(todo) * len(a.seeds)
    k = 0
    t0 = time.time()
    for seed in a.seeds:
        for arm in todo:
            k += 1
            if (arm["arm"], seed) in have:
                print(f"[{k}/{total}] skip {arm['arm']} s{seed} (done)", flush=True)
                continue
            cmd = [PY, "train.py", "--dataset", "cifar10", "--arch", "resnet",
                   "--epochs", str(a.epochs), "--lr", str(a.lr),
                   "--batch-size", str(a.batch_size), "--workers", str(a.workers),
                   "--seed", str(seed), "--ema-mode", arm["ema"],
                   "--snap-every", str(arm["snap"]), "--quant-mode", arm["quant"],
                   "--amp", "--grad-clip", "1.0", "--ema-decay", "0.99"]
            log = Path("logs") / f"cifar_{arm['arm'].replace('@', '')}_s{seed}.log"
            t1 = time.time()
            print(f"[{k}/{total}] {arm['arm']} s{seed} -> {log}", flush=True)
            with log.open("w", encoding="utf-8", errors="replace") as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
            if rc != 0:
                print(f"FAILED rc={rc} {arm['arm']} s{seed}; 见 {log}", flush=True)
                raise SystemExit(rc)
            run_dir = ""
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("run dir: "):
                    run_dir = line[len("run dir: "):].strip()
            if not run_dir:      # 老版本 train.py 不打这一行时按时间戳兜底
                cands = sorted(Path("results").glob(f"cifar-*-s{seed}-*"),
                               key=lambda p: p.stat().st_mtime)
                run_dir = str(cands[-1]) if cands else ""
            with STATE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"arm": arm["arm"], "seed": seed, "dir": run_dir}) + "\n")
            print(f"    done in {time.time() - t1:.0f}s -> {run_dir} "
                  f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    print("QUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
