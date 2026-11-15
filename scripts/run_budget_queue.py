"""训练预算 / 学习率边界的两条补实验队列（第六十一轮）。

回答评审问题：“Is the cost an artifact of a 20-epoch budget and a small learning rate?”
  * 60 epoch, lr 1e-4：补 seed 3..7 的 N=1 / N=off（seed 0..2 已在 results_long）
  * 20 epoch, lr 3e-3：补 seed 3..7 的 N=1 / N=20 / N=off（seed 0..2 已在 results_lr）

统一口径：--split-seed 0（固定划分，与主实验一致）、--ema-mode none、
--head relu_qat、--quant-mode overwrite、--val-windows 2000。

    python scripts/run_budget_queue.py --dry-run
    python scripts/run_budget_queue.py

断点台账 logs/_budget_done.txt，每行 epochs,lr,out,seed,snap。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PY = "C:/Users/USER/.conda/envs/e2e/python.exe"
ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "logs" / "_budget_done.txt"

FIXED = ["--head", "relu_qat", "--ema-mode", "none", "--ema-decay", "0.99",
         "--batch-size", "64", "--amp", "--grad-clip", "1.0", "--workers", "6",
         "--quant-mode", "overwrite", "--split-seed", "0",
         "--raw", "data/utkface_raw224.npy", "--val-windows", "2000"]


def jobs(seeds) -> list[dict]:
    out = []
    for seed in seeds:
        for snap in ("1", "0"):
            out.append({"seed": seed, "epochs": "60", "lr": "1e-4",
                        "snap": snap, "out": "results_long"})
        for snap in ("1", "20", "0"):
            out.append({"seed": seed, "epochs": "20", "lr": "3e-3",
                        "snap": snap, "out": "results_lr"})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="3,4,5,6,7")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    seeds = [x.strip() for x in a.seeds.split(",") if x.strip()]
    todo = jobs(seeds)
    print("queue: %d runs%s" % (len(todo), " (dry-run)" if a.dry_run else ""), flush=True)
    done = set()
    if LEDGER.exists():
        done = {ln.strip() for ln in LEDGER.read_text(encoding="utf-8").splitlines() if ln.strip()}
    for i, j in enumerate(todo, 1):
        cell = "%s,%s,%s,%s,%s" % (j["epochs"], j["lr"], j["out"], j["seed"], j["snap"])
        if a.skip_existing and cell in done:
            print("[%d/%d] skip (done): %s" % (i, len(todo), cell), flush=True)
            continue
        cmd = [PY, "train.py", *FIXED, "--epochs", j["epochs"], "--lr", j["lr"],
               "--seed", str(j["seed"]), "--snap-every", j["snap"], "--out", j["out"]]
        print("[%d/%d] %s" % (i, len(todo), " ".join(cmd)), flush=True)
        if a.dry_run:
            continue
        rc = 1
        for attempt in range(3):
            t0 = time.time()
            rc = subprocess.call(cmd, cwd=str(ROOT))
            if rc == 0:
                break
            print("RETRY %d/3 rc=%d after %.0fs: %s" % (attempt + 1, rc, time.time() - t0, cell), flush=True)
            time.sleep(30)
        if rc != 0:
            print("FAILED rc=%d after 3 attempts: %s" % (rc, cell), flush=True)
            sys.exit(rc if rc > 0 else 1)
        print("ok %.0fs" % (time.time() - t0), flush=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(cell + "\n")
        done.add(cell)
    print("DONE queue (%d runs)" % len(todo), flush=True)


if __name__ == "__main__":
    main()
