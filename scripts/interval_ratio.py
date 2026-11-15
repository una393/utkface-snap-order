"""Normalise the interval-1 cost by the projection-off error.

Section 5.5 reports six external checks and calls five of them directional. The
comparison that does transfer across settings is a ratio: how much MAE the
interval-1 overwrite adds, divided by the error the same model has with
projection disabled. That is dimensionless, so UTKFace age regression, KonIQ-10k
image quality and CIFAR-10 classification can be put side by side, and a new
setting either lands inside the band or falsifies it.

Paired per seed: ratio_i = (err_i(interval 1) - err_i(off)) / err_i(off), with the
95 % interval from the t distribution over seeds (df = n - 1). CIFAR-10 uses
top-1 error rate in place of MAE; the two are both bounded error measures and the
ratio is read the same way.

Writes quant_runs/interval_ratio.csv.
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from pathlib import Path

T975 = {7: 2.3646, 3: 3.1824, 4: 2.7764}


def t975(df: int) -> float:
    return T975.get(df, 1.96)


def ratio_row(setting: str, task: str, off: list[float], on: list[float]) -> dict:
    assert len(off) == len(on) and len(off) >= 2, (setting, len(off), len(on))
    r = [(a - b) / b for a, b in zip(on, off)]
    n = len(r)
    m = st.mean(r)
    sd = st.stdev(r)
    half = t975(n - 1) * sd / math.sqrt(n)
    return {
        "setting": setting,
        "task": task,
        "n_seeds": n,
        "off_error": round(st.mean(off), 6),
        "interval1_error": round(st.mean(on), 6),
        "cost": round(st.mean(on) - st.mean(off), 6),
        "ratio": round(m, 4),
        "ratio_ci_lo": round(m - half, 4),
        "ratio_ci_hi": round(m + half, 4),
        "seeds_ratio_positive": sum(1 for x in r if x > 0),
    }


def main() -> None:
    rows: list[dict] = []

    # UTKFace, main sequence: projection off is none/N=0, interval 1 is none/N=1.
    recs = [r for r in csv.DictReader(open("quant_runs/heldout_eval_v3.csv", encoding="utf-8"))
            if r["subset"] == "heldout_1556" and r["mode"] == "none" and r["N"] in ("0", "1")]
    by = {(r["seed"], r["N"]): float(r["mae_fp32"]) for r in recs}
    seeds = sorted({r["seed"] for r in recs if (r["seed"], "0") in by and (r["seed"], "1") in by})
    rows.append(ratio_row("UTKFace, RegNetY-008 (main sequence)",
                          "age regression (MAE)",
                          [by[(s, "0")] for s in seeds],
                          [by[(s, "1")] for s in seeds]))

    # ResNet-18 architecture axis at 1e-3, the converged axis quoted in the text.
    for path, label in (("quant_runs/ext_deploy_arch_heldout.csv", "UTKFace, ResNet-18 (architecture axis, 1e-3)"),
                        ("quant_runs/ext_deploy_arch_b_end_heldout.csv", "UTKFace, ResNet-18 (architecture axis, 1e-4)")):
        recs = [r for r in csv.DictReader(open(path, encoding="utf-8"))
                if r["mode"] == "none" and r["N"] in ("0", "1")]
        by = {(r["seed"], r["N"]): float(r["mae_p32"]) for r in recs}
        seeds = sorted({r["seed"] for r in recs if (r["seed"], "0") in by and (r["seed"], "1") in by})
        rows.append(ratio_row(label, "age regression (MAE)",
                              [by[(s, "0")] for s in seeds],
                              [by[(s, "1")] for s in seeds]))

    # KonIQ-10k data axis.
    recs = [r for r in csv.DictReader(open("quant_runs/ext_deploy_koniq_heldout.csv", encoding="utf-8"))
            if r["mode"] == "none" and r["N"] in ("0", "1")]
    by = {(r["seed"], r["N"]): float(r["mae_p32"]) for r in recs}
    seeds = sorted({r["seed"] for r in recs if (r["seed"], "0") in by and (r["seed"], "1") in by})
    rows.append(ratio_row("KonIQ-10k (data axis)",
                          "image quality (MAE)",
                          [by[(s, "0")] for s in seeds],
                          [by[(s, "1")] for s in seeds]))

    # CIFAR-10 classification arm, top-1 error rate.
    recs = list(csv.DictReader(open("quant_runs/cifar_clean_split.csv", encoding="utf-8")))
    by = {(r["seed"], r["arm"]): float(r["test_err_live"]) for r in recs}
    seeds = sorted({r["seed"] for r in recs if (r["seed"], "live@inf") in by and (r["seed"], "live@1") in by})
    rows.append(ratio_row("CIFAR-10, ResNet-18 (task axis)",
                          "10-class (top-1 error)",
                          [by[(s, "live@inf")] for s in seeds],
                          [by[(s, "live@1")] for s in seeds]))

    out = Path("quant_runs/interval_ratio.csv")
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("%-42s ratio %.3f  [%.3f, %.3f]  n=%d  off %.4f  cost %.4f" %
              (r["setting"], r["ratio"], r["ratio_ci_lo"], r["ratio_ci_hi"],
               r["n_seeds"], r["off_error"], r["cost"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
