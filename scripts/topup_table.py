"""Before/after view of the run top-ups that happened after the analysis plan was fixed.

Every confirmatory contrast was first read on the initial seeds (0, 1, 2) and re-read
after the seed top-up (0-7).  This script recomputes both readings from the same
artifacts and writes quant_runs/topup_table.csv, so that the disclosure in the paper
is reproducible rather than quoted.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HELD = ROOT / "quant_runs" / "heldout_eval_v3.csv"
STE = ROOT / "quant_runs" / "ste_v3.csv"
BUDGET = ROOT / "quant_runs" / "budget_contrast.json"
OUT = ROOT / "quant_runs" / "topup_table.csv"

EARLY = ("0", "1", "2")


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
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
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def p_two_sided(diffs):
    n = len(diffs)
    if n < 2:
        return float("nan")
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    if sd == 0.0:
        return 0.0
    t = abs(m) / (sd / math.sqrt(n))
    df = n - 1
    return _betai(df / 2.0, 0.5, df / (df + t * t))


MODE = {"none": "live", "before_snap": "ema_before", "after_snap": "ema_after"}


def cell_table():
    cells = {}
    with open(HELD, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["subset"] != "heldout_1556":
                continue
            mode = MODE[r["mode"]]
            cells.setdefault((mode, r["N"]), {})[r["seed"]] = float(r["mae_fp32"])
    return cells


def ste_table():
    cells = {}
    with open(STE, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["subset"] != "heldout_1556":
                continue
            cells.setdefault((r["arm"], "8bit"), {})[r["seed"]] = float(r["mae_8"])
    return cells


def paired(cells, a, b):
    seeds = sorted(set(cells[a]) & set(cells[b]), key=int)
    return seeds, [cells[a][s] - cells[b][s] for s in seeds]


def summarise(diffs, seeds):
    n = len(diffs)
    m = sum(diffs) / n
    pos = sum(1 for d in diffs if d > 0)
    return m, pos, n, p_two_sided(diffs)


def row(label, basis, unit, seeds, diffs):
    early = [diffs[i] for i, s in enumerate(seeds) if s in EARLY]
    m_e, pos_e, n_e, _ = summarise(early, seeds[: len(early)])
    m_f, pos_f, n_f, p_f = summarise(diffs, seeds)
    return {
        "contrast": label,
        "basis": basis,
        "unit": unit,
        "seeds_early": f"{EARLY[0]}-{EARLY[-1]} (n = {n_e})",
        "est_early": round(m_e, 6),
        "signs_early": f"{pos_e}/{n_e} positive",
        "seeds_final": f"{seeds[0]}-{seeds[-1]} (n = {n_f})",
        "est_final": round(m_f, 6),
        "signs_final": f"{pos_f}/{n_f} positive",
        "change": round(m_f - m_e, 6),
        "p_final": float(f"{p_f:.3g}"),
    }


cells = cell_table()
ste = ste_table()
out = []

seeds, d = paired(cells, ("live", "1"), ("live", "0"))
out.append(row("interval 1 against projection off (contrast a)", "held-out 1,556", "run", seeds, d))
seeds, d = paired(cells, ("live", "20"), ("live", "0"))
out.append(row("interval 20 against projection off (contrast b)", "held-out 1,556", "run", seeds, d))
seeds, d = paired(cells, ("ema_after", "20"), ("live", "20"))
out.append(row("ema_after against live at interval 20 (contrast c)", "held-out 1,556", "run", seeds, d))
seeds, d = paired(cells, ("ema_before", "20"), ("ema_after", "20"))
out.append(row("ema_before against ema_after at interval 20 (contrast d)", "held-out 1,556", "run", seeds, d))
seeds, d = paired(ste, ("ste", "8bit"), ("overwrite", "8bit"))
out.append(row("STE against in-place overwriting", "held-out 1,556", "run", seeds, d))
seeds, d = paired(ste, ("ste", "8bit"), ("no_projection", "8bit"))
out.append(row("STE against projection off", "held-out 1,556", "run", seeds, d))

budget = json.loads(BUDGET.read_text(encoding="utf-8"))
for key, label in (
    ("long60:N1_minus_N0", "interval 1 against projection off, 60-epoch budget"),
    ("lr3e-3:N1_minus_N0", "interval 1 against projection off, 30x learning rate"),
):
    entry = budget["contrasts"][key]
    block = entry["fp32"]
    seeds = [str(s) for s in entry["seeds"]]
    out.append(row(label, "held-out 1,556", "run", seeds, block["diffs"]))

cols = list(out[0])
with open(OUT, "w", newline="\n", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(out)

for r in out:
    print(f'{r["contrast"]:<58} early {r["est_early"]:+.6f} ({r["signs_early"]:<15}) '
          f'final {r["est_final"]:+.6f} ({r["signs_final"]:<16}) change {r["change"]:+.6f} p {r["p_final"]}')
print("wrote", OUT.relative_to(ROOT))
