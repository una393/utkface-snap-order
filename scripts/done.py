"""判断某个 (head, ema, seed) 是否已有完成的 run（epochs 达标且 result.json 存在）。"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True)
    p.add_argument("--ema", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--min-epochs", type=int, default=20)
    a = p.parse_args()

    for hp in glob.glob("results/*/history.csv"):
        run = Path(hp).parent
        rj = run / "result.json"
        if not rj.exists():
            continue
        try:
            args = json.loads(rj.read_text())["args"]
        except Exception:
            continue
        if (args.get("head"), args.get("ema_mode"), args.get("seed")) != (a.head, a.ema, a.seed):
            continue
        rows = list(csv.DictReader(open(hp, newline="")))
        if len(rows) >= a.min_epochs:
            print(f"done: {run.name}")
            sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
