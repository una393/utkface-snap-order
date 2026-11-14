"""汇总 results/：最佳轮、最终轮、是否跑赢零基线、是否收敛。"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path


def main() -> None:
    runs = []
    for p in glob.glob("results/*/history.csv"):
        run = Path(p).parent
        res = json.loads((run / "result.json").read_text())
        rows = list(csv.DictReader(open(p, newline="")))
        live = [float(r["val_mae_live"]) for r in rows]
        ema = [float(r["val_mae_ema"]) for r in rows]
        runs.append({"run": run.name, "head": res["args"]["head"],
                     "ema": res["args"]["ema_mode"], "ep": len(rows),
                     "best": min(live), "best_ep": live.index(min(live)), "last": live[-1],
                     "ema_best": min(ema), "zero": res["zero_baseline_mae"]})
    runs.sort(key=lambda r: (r["head"], r["ema"]))
    print(f"{'head':13s} {'ema':12s} {'ep':>3s} {'zero':>7s} {'best_live':>9s} "
          f"{'@ep':>4s} {'last':>7s} {'best_ema':>8s}  verdict")
    for r in runs:
        v = "BEATS" if r["best"] < r["zero"] else "FAIL"
        print(f"{r['head']:13s} {r['ema']:12s} {r['ep']:>3d} {r['zero']:>7.4f} "
              f"{r['best']:>9.4f} {r['best_ep']:>4d} {r['last']:>7.4f} {r['ema_best']:>8.4f}  {v}")


if __name__ == "__main__":
    main()
