"""Subgroup deployment MAE by age band, sex and race on the 1,556-image basis.

Reads the cached per-image errors in quant_runs/leakpred/<run>__{fp32,8bit}.npz
(written by scripts/leak_contrast.py) and the subgroup labels encoded in the
UTKFace file names (age_gender_race_...jpg).
"""
import csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from data.utkface import scan
from scripts import heldout_eval as he

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SP = 0
SEEDS = list(range(8))
# (label, ema_mode, interval) -- ema_mode as stored in the run inventory
ARMS = {"live1": ("none", 1), "live20": ("none", 20), "off": ("none", 0), "ema20": ("after_snap", 20)}
RACE = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
AGE_BINS = [(0, 19, "<20"), (20, 39, "20-39"), (40, 59, "40-59"), (60, 200, "60+")]

recs = scan("D:/data/utkface/UTKFace")
first, mon, held, val, mon_names, inter = he.build_views(recs, SP)
assert len(held) == 1556, len(held)

names = [Path(r["path"]).name for r in held]
age = np.array([int(n.split("_")[0]) for n in names])
sex = np.array([int(n.split("_")[1]) for n in names])
race = np.array([int(n.split("_")[2]) for n in names])

tab = list(csv.DictReader(open("quant_runs/heldout_eval_v3.csv", encoding="utf-8")))
ref = {(r["mode"], int(r["N"]), int(r["seed"])): r["run"] for r in tab if r["subset"] == "heldout_1556"}
cache = Path("quant_runs/leakpred")


def err(arm, seed, bits):
    mode, N = ARMS[arm]
    run = ref[(mode, N, seed)]
    p = cache / ("%s__%s.npz" % (run, bits))
    if not p.exists():
        raise SystemExit("missing cache %s (run scripts/leak_contrast.py)" % p)
    return np.load(p)["err"]


def arm_mae(arm, bits, mask):
    return float(np.mean([err(arm, s, bits)[mask].mean() for s in SEEDS]))


groups = [("age", lab, (age >= lo) & (age <= hi)) for lo, hi, lab in AGE_BINS]
groups += [("sex", "male", sex == 0), ("sex", "female", sex == 1)]
groups += [("race", lab, race == k) for k, lab in RACE.items()]

rows = []
for family, lab, m in groups:
    n = int(m.sum())
    if n < 20:
        continue
    fp32_20 = arm_mae("live20", "fp32", m)
    q8_20 = arm_mae("live20", "8bit", m)
    fp32_1 = arm_mae("live1", "fp32", m)
    fp32_off = arm_mae("off", "fp32", m)
    rows.append(dict(family=family, group=lab, n=n, fp32_20=fp32_20, q8_20=q8_20,
                     d8=q8_20 - fp32_20, cost=fp32_1 - fp32_off,
                     ratio=fp32_20 / max(fp32_off, 1e-9)))

with open("quant_runs/subgroup_table.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["family", "group", "n", "fp32_20", "q8_20", "d8", "cost", "ratio"])
    w.writeheader()
    for r in rows:
        w.writerow({k: ("%.6f" % v if isinstance(v, float) else v) for k, v in r.items()})

print("%-6s %-8s %5s %8s %8s %9s %9s %6s" % ("family", "group", "n", "fp32@20", "8bit@20", "d8-fp32", "cost@1", "rel"))
for r in rows:
    print("%-6s %-8s %5d %8.4f %8.4f %+9.4f %+9.4f %6.2f" % (
        r["family"], r["group"], r["n"], r["fp32_20"], r["q8_20"], r["d8"], r["cost"], r["ratio"]))

allm = np.ones(len(held), dtype=bool)
print("%-6s %-8s %5d %8.4f %8.4f %+9.4f %+9.4f %6.2f" % (
    "all", "-", len(held), arm_mae("live20", "fp32", allm), arm_mae("live20", "8bit", allm),
    arm_mae("live20", "8bit", allm) - arm_mae("live20", "fp32", allm),
    arm_mae("live1", "fp32", allm) - arm_mae("off", "fp32", allm),
    arm_mae("live20", "fp32", allm) / arm_mae("off", "fp32", allm)))
