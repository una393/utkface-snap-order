"""同学习率 / 中间间隔的外推队列（第三十六轮，架构轴补 lr 1e-4、数据轴补间隔 20）。

与 scripts/run_ext_queue.py 的差别只有三处，旧脚本不动：
  * --lr 显式给，不写死在 AXIS_EXTRA 里；
  * --snaps 可给多个间隔（旧脚本固定 1,0）；
  * --out 落到独立目录，避免与 results/ 里的主实验 glob 互相污染。

    python scripts/run_ext_lr_queue.py --axis arch --lr 1e-4 --snaps 1,20,0 \
        --seeds 0,1,2,3,4,5,6,7 --out results_ext_lr1e4

断点台账 logs/_ext_lr_done.txt，每行 axis,lr,out,seed,mode,snap。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PY = "C:/Users/24399/.conda/envs/e2e/python.exe"
ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "logs" / "_ext_lr_done.txt"

FIXED = ["--epochs", "20", "--batch-size", "64", "--amp",
         "--grad-clip", "1.0", "--ema-decay", "0.99", "--workers", "6",
         "--quant-mode", "overwrite", "--head", "relu_qat"]

AXIS_EXTRA = {
    "arch": ["--arch", "resnet", "--raw", "data/utkface_raw224.npy", "--val-windows", "2000"],
    "data": ["--dataset", "koniq", "--arch", "net", "--val-windows", "1110"],
}


def jobs(axis: str, seeds, modes, snaps, lr: str, out: str):
    res = []
    for seed in seeds:
        for mode in modes:
            for snap in snaps:
                res.append([PY, "train.py", *FIXED, "--seed", str(seed),
                            "--ema-mode", mode, "--snap-every", str(snap),
                            "--lr", lr, "--out", out, *AXIS_EXTRA[axis]])
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["arch", "data"], required=True)
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--modes", default="none,after_snap")
    p.add_argument("--snaps", default="1,20,0")
    p.add_argument("--lr", default="1e-4")
    p.add_argument("--out", default="results_ext_lr1e4")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    seeds = [x.strip() for x in a.seeds.split(",") if x.strip()]
    snaps = [x.strip() for x in a.snaps.split(",") if x.strip()]

    modes = [x.strip() for x in a.modes.split(",") if x.strip()]
    cmds = jobs(a.axis, seeds, modes, snaps, a.lr, a.out)
    print("axis=%s lr=%s out=%s: %d runs%s"
          % (a.axis, a.lr, a.out, len(cmds), " (dry-run)" if a.dry_run else ""), flush=True)
    done = set()
    if LEDGER.exists():
        done = {ln.strip() for ln in LEDGER.read_text(encoding="utf-8").splitlines() if ln.strip()}
    for i, cmd in enumerate(cmds, 1):
        line = " ".join(cmd)
        print("[%d/%d] %s" % (i, len(cmds), line), flush=True)
        if a.dry_run:
            continue
        cell = "%s,%s,%s,%s,%s,%s" % (a.axis, a.lr, a.out,
                                      cmd[cmd.index("--seed") + 1],
                                      cmd[cmd.index("--ema-mode") + 1],
                                      cmd[cmd.index("--snap-every") + 1])
        if a.skip_existing and cell in done:
            print("skip (done): %s" % cell, flush=True)
            continue
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print("FAILED rc=%d after %.0fs: %s" % (rc, time.time() - t0, line), flush=True)
            sys.exit(rc if rc > 0 else 1)
        print("ok %.0fs" % (time.time() - t0), flush=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(cell + "\n")
        done.add(cell)
    print("DONE axis=%s (%d runs)" % (a.axis, len(cmds)), flush=True)


if __name__ == "__main__":
    main()
