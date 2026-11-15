"""按控制头汇总多个 seed：mean ± std，判断"持平"是否成立。"""
from __future__ import annotations

import csv
import glob
import json
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
from pathlib import Path


def main() -> None:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for p in glob.glob("results/*/history.csv"):
        run = Path(p).parent
        if not (run / "result.json").exists():   # 正在跑的 run 先跳过
            continue
        res = json.loads((run / "result.json").read_text())
        rows = list(csv.DictReader(open(p, newline="")))
        if len(rows) < 20:
            continue
        live = [float(r["val_mae_live"]) for r in rows]
        groups[(res["args"]["head"], res["args"]["ema_mode"])].append(min(live))

    print(f"{'head':14s} {'ema':12s} {'n':>2s}  mean ± std        values")
    print("-" * 78)
    for (head, ema), vals in sorted(groups.items()):
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{head:14s} {ema:12s} {len(vals):>2d}  {m:.4f} ± {s:.4f}   "
              f"{[round(v, 4) for v in vals]}")

    print()
    base = [h for h in groups if h[1] == "after_snap"]
    means = {h[0]: statistics.mean(groups[h]) for h in base}
    if len(means) == 3:
        lo, hi = min(means.values()), max(means.values())
        spread = hi - lo
        print(f"三个头 best_live 的均值范围 {lo:.4f}–{hi:.4f}，极差 {spread:.4f}")
        pooled = statistics.mean(s for h in base for s in
                                 [statistics.stdev(groups[h]) if len(groups[h]) > 1 else 0.0])
        print(f"组内 seed 标准差均值 {pooled:.4f}")
        print("判据：极差 ≤ 组内标准差 → 可称'持平'；极差远大于组内标准差 → 有真实差异。")
        print("结论：", "持平成立" if spread <= max(pooled, 1e-4) * 1.5 else "存在可测差异")


if __name__ == "__main__":
    main()
