
"""Order effect at interval 20: per-seed readings, Welch CI, and a TOST at margin 0.0018.

Reads the six main-configuration runs (before_snap / after_snap x seeds 0,1,2 at
interval 20) and reports the shadow validation MAE of each, the Welch interval for
before - after, and the two one-sided tests of |Delta| >= 0.0018.

Usage: D:/tool/Python/python.exe scripts/order_tost.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

MARGIN = 0.0018
N20 = 20
RUNS = {
    "before_snap": [
        "utk-relu_qat-before_snap-s0-1793032011",
        "utk-relu_qat-before_snap-s1-1793037047",
        "utk-relu_qat-before_snap-s2-1793039497",
    ],
    "after_snap": [
        "utk-relu_qat-after_snap-s0-1793033217",
        "utk-relu_qat-after_snap-s1-1793038259",
        "utk-relu_qat-after_snap-s2-1793040741",
    ],
}


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(p: float, df: float) -> float:
    lo, hi = 0.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def welch(a: list[float], b: list[float]) -> dict:
    v1, v2 = st.variance(a), st.variance(b)
    se = math.sqrt(v1 / len(a) + v2 / len(b))
    df = (v1 / len(a) + v2 / len(b)) ** 2 / (
        (v1 / len(a)) ** 2 / (len(a) - 1) + (v2 / len(b)) ** 2 / (len(b) - 1)
    )
    diff = st.mean(a) - st.mean(b)
    tc = t_ppf(0.975, df)
    return {
        "diff": diff,
        "se": se,
        "df": df,
        "lo": diff - tc * se,
        "hi": diff + tc * se,
        # TOST: H0 |diff| >= margin, rejected when both one-sided tests are significant
        "p_low": 1.0 - t_cdf((diff + MARGIN) / se, df),
        "p_high": 1.0 - t_cdf((MARGIN - diff) / se, df),
    }


def main() -> int:
    vals = {}
    for mode, runs in RUNS.items():
        row = []
        for name in runs:
            path = Path("results") / name / "result.json"
            if not path.exists():
                print("MISSING", path, file=sys.stderr)
                return 1
            row.append(float(json.loads(path.read_text(encoding="utf-8"))["val_mae_ema"]))
        vals[mode] = row
        print(
            f"{mode:12s} best-shadow val MAE "
            + ", ".join(f"{v:.6f}" for v in row)
            + f"  mean {st.mean(row):.6f}  sd {st.stdev(row):.6f}"
        )

    res = welch(vals["before_snap"], vals["after_snap"])
    print(
        "Welch (shadow val MAE): diff {diff:+.6f}  se {se:.6f}  df {df:.2f}  "
        "95% CI [{lo:+.6f}, {hi:+.6f}]".format(**res)
    )
    p_tost = max(res["p_low"], res["p_high"])
    print(
        f"TOST margin {MARGIN}: p_low {res['p_low']:.4f}  p_high {res['p_high']:.4f}  "
        f"p_tost {p_tost:.4f} -> " + ("equivalent" if p_tost < 0.05 else "NOT equivalent")
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
