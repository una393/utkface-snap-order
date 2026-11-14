"""Distribution-free checks on the six confirmatory contrasts.

The plan fixed a paired t test with Holm correction.  This script adds three checks that
make no normality assumption: the exact Wilcoxon signed-rank test over the 2^n sign
assignments of the eight seed pairs, a percentile bootstrap interval over the same pairs,
and the standardised paired effect size d_z.  Nothing here replaces the pre-registered
test; the numbers go to the Supplementary Material as a robustness table.
"""
import csv
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topup_table as tt  # noqa: E402  (loaders and the paired t test are shared)

ROOT = HERE.parent
OUT = ROOT / "quant_runs" / "contrast_checks.csv"
RESAMPLES = 10000
BOOT_SEED = 20261109


def wilcoxon_exact(diffs):
    """Two-sided exact Wilcoxon signed-rank p over all 2^n sign assignments."""
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        share = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = share
        i = j + 1
    obs = sum(ranks[i] for i in range(n) if nz[i] > 0)
    stat = abs(obs - n * (n + 1) / 4.0)
    hits = 0
    for mask in range(1 << n):
        w = sum(ranks[i] for i in range(n) if mask >> i & 1)
        if abs(w - n * (n + 1) / 4.0) >= stat - 1e-12:
            hits += 1
    return hits / float(1 << n)


def bootstrap_ci(diffs, alpha=0.05):
    rng = random.Random(BOOT_SEED)
    n = len(diffs)
    means = []
    for _ in range(RESAMPLES):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(math.floor(alpha / 2 * RESAMPLES))]
    hi = means[int(math.ceil((1 - alpha / 2) * RESAMPLES)) - 1]
    return lo, hi


def dz(diffs):
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    return m / sd if sd else float("nan")


cells = tt.cell_table()
ste = tt.ste_table()
rows = []

SPEC = [
    ("a", "interval 1 minus projection off", cells, ("live", "1"), ("live", "0")),
    ("b", "interval 20 minus projection off", cells, ("live", "20"), ("live", "0")),
    ("c", "ema_after minus live at interval 20", cells, ("ema_after", "20"), ("live", "20")),
    ("d", "ema_before minus ema_after at interval 20", cells, ("ema_before", "20"), ("ema_after", "20")),
    ("e", "STE minus in-place overwriting", ste, ("ste", "8bit"), ("overwrite", "8bit")),
    ("f", "STE minus projection off", ste, ("ste", "8bit"), ("no_projection", "8bit")),
]

for tag, label, table, left, right in SPEC:
    seeds, diffs = tt.paired(table, left, right)
    n = len(diffs)
    m = sum(diffs) / n
    pos = sum(1 for d in diffs if d > 0)
    p_t = tt.p_two_sided(diffs)
    p_w = wilcoxon_exact(diffs)
    lo, hi = bootstrap_ci(diffs)
    rows.append({
        "contrast": tag,
        "definition": label,
        "n": n,
        "estimate": round(m, 6),
        "signs": "%d/%d positive" % (pos, n),
        "d_z": round(dz(diffs), 3),
        "p_paired_t": float("%.3g" % p_t),
        "p_wilcoxon_exact": float("%.3g" % p_w),
        "boot_lo": round(lo, 6),
        "boot_hi": round(hi, 6),
    })

with open(OUT, "w", newline="\n", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

for r in rows:
    print("%s  %+.6f  d_z %+.3f  t p %.3g  Wilcoxon p %.3g  boot [%+.6f, %+.6f]  %s"
          % (r["contrast"], r["estimate"], r["d_z"], r["p_paired_t"], r["p_wilcoxon_exact"],
             r["boot_lo"], r["boot_hi"], r["signs"]))
print("wrote", OUT.relative_to(ROOT))
