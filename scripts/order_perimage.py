"""Per-image view of the projection/averaging order contrast.

The mean error of the two orders is the same to 0.00003 MAE on the 1,556-image
basis, which the manuscript reported as a flat basin it had not measured. This
script measures what the mean hides: how far the error of a single image moves
when the order of projection and averaging is swapped, using the cached per-image
errors in quant_runs/leakpred (one array of 1,556 absolute errors per run and bit
width). Nothing is re-evaluated and no GPU is touched.

Writes quant_runs/order_perimage.csv.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

SRC = Path("quant_runs/leakpred")
NAME = re.compile(r"utk-relu_qat-(?P<mode>none|before_snap|after_snap)-s(?P<seed>\d+)-\d+__(?P<bits>fp32|8bit)\.npz")
FIELDS = ["basis", "seed", "run_before", "run_after", "mae_before", "mae_after", "d_before_after",
          "pearson_r", "sd_d", "share_d_pos", "mean_abs_d", "median_abs_d", "max_abs_d",
          "share_gt_0.005", "share_gt_0.01"]


def load() -> dict:
    out = {}
    for p in sorted(SRC.glob("*.npz")):
        m = NAME.match(p.name)
        if not m:
            continue
        out[(m["mode"], int(m["seed"]), m["bits"])] = (p.stem.rsplit("__", 1)[0], np.load(p)["err"])
    return out


def main() -> None:
    errs = load()
    seeds = sorted({s for (mode, s, b) in errs if mode == "before_snap" and b == "fp32"})
    rows, pooled = [], {}
    for bits in ("fp32", "8bit"):
        for s in seeds:
            rb, eb = errs[("before_snap", s, bits)]
            ra, ea = errs[("after_snap", s, bits)]
            d = eb - ea
            row = {
                "basis": bits, "seed": s, "run_before": rb, "run_after": ra,
                "mae_before": round(float(eb.mean()), 6), "mae_after": round(float(ea.mean()), 6),
                "d_before_after": round(float(d.mean()), 6),
                "pearson_r": round(float(np.corrcoef(eb, ea)[0, 1]), 4),
                "sd_d": round(float(d.std(ddof=1)), 5),
                "share_d_pos": round(float((d > 0).mean()), 4),
                "mean_abs_d": round(float(np.abs(d).mean()), 5),
                "median_abs_d": round(float(np.median(np.abs(d))), 5),
                "max_abs_d": round(float(np.abs(d).max()), 5),
                "share_gt_0.005": round(float((np.abs(d) > 0.005).mean()), 4),
                "share_gt_0.01": round(float((np.abs(d) > 0.01).mean()), 4),
            }
            rows.append(row)
            pooled.setdefault(bits, []).append(row)

    out = Path("quant_runs/order_perimage.csv")
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print("pairs: %d seeds x 2 bit widths" % len(seeds))
    for bits, rs in pooled.items():
        def rng(k):
            v = [r[k] for r in rs]
            return min(v), max(v)
        d = [r["d_before_after"] for r in rs]
        print()
        print("[%s] d(before-after) mean %.6f  range %.6f .. %.6f  negative %d/%d"
              % (bits, float(np.mean(d)), min(d), max(d), sum(1 for x in d if x < 0), len(d)))
        print("  pearson r        %.4f .. %.4f" % rng("pearson_r"))
        print("  sd(d)            %.5f .. %.5f" % rng("sd_d"))
        print("  share d>0        %.4f .. %.4f" % rng("share_d_pos"))
        print("  mean |d|         %.5f .. %.5f" % rng("mean_abs_d"))
        print("  median |d|       %.5f .. %.5f" % rng("median_abs_d"))
        print("  max |d|          %.5f .. %.5f" % rng("max_abs_d"))
        print("  share |d|>0.005  %.4f .. %.4f" % rng("share_gt_0.005"))
        print("  share |d|>0.01   %.4f .. %.4f" % rng("share_gt_0.01"))
    print()
    print("wrote", out)


if __name__ == "__main__":
    main()
