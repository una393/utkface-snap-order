"""3x3 交叉设计缺的四格补跑队列（split_seed, seed) = (1,2)(1,3)(2,1)(2,3)。

固定配置与间隔 20 的主配置臂逐项一致（after_snap / gamma 0.99 / 20 epochs / snap 20 /
quant-mode overwrite）。(1,1) 与 (2,2) 已是主配置 run，不在此列。

已完成的格会被跳过（按 results/*/result.json 的 args 判重，目录名不含 split_seed），因此可重复启动。
任一训练子进程 returncode != 0 即抛错退出，不把失败当成功。

用法: python scripts/run_split_cross.py [--dry-run]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
BASE = [PY, "-u", "train.py", "--head", "relu_qat", "--ema-mode", "after_snap",
        "--ema-decay", "0.99", "--epochs", "20", "--batch-size", "64", "--lr", "1e-4",
        "--amp", "--grad-clip", "1.0", "--workers", "6", "--raw", "data/utkface_raw224.npy",
        "--quant-mode", "overwrite", "--snap-every", "20"]
CELLS = [(1, 2), (1, 3), (2, 1), (2, 3)]


def done_set() -> set[tuple[int, int]]:
    out = set()
    for rj in Path("results").glob("utk-relu_qat-*/result.json"):
        try:
            a = json.loads(rj.read_text(encoding="utf-8")).get("args", {})
        except Exception:
            continue
        if "-ste-" in rj.parent.name:
            continue
        if a.get("head") != "relu_qat" or a.get("ema_mode") != "after_snap":
            continue
        if a.get("epochs") != 20 or a.get("snap_every") != 20 \
                or a.get("quant_mode", "overwrite") != "overwrite" \
                or round(float(a.get("ema_decay", 0.99)), 4) != 0.99:
            continue
        seed = int(a.get("seed", 0))
        sp = a.get("split_seed", -1)
        sp = seed if sp is None or int(sp) < 0 else int(sp)
        out.add((int(sp), seed))
    return out


def main() -> None:
    dry = "--dry-run" in sys.argv
    already = done_set()
    for sp, seed in CELLS:
        cmd = BASE + ["--split-seed", str(sp), "--seed", str(seed)]
        if (sp, seed) in already:
            print("[X] SKIP split=%d seed=%d (already present)" % (sp, seed), flush=True)
            continue
        print("[X] >>> split=%d seed=%d :: %s" % (sp, seed, " ".join(cmd[3:])), flush=True)
        if dry:
            continue
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        dt = time.time() - t0
        tail = (p.stdout or "").strip().splitlines()[-4:]
        print("[X] <<< split=%d seed=%d rc=%d %.0fs" % (sp, seed, p.returncode, dt), flush=True)
        print("\n".join(tail) or (p.stderr or "")[-800:], flush=True)
        if p.returncode != 0:
            raise SystemExit("[X] FAILED split=%d seed=%d rc=%d; 中止队列"
                             % (sp, seed, p.returncode))
    print("[X] QUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
