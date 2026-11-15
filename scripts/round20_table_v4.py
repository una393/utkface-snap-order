# -*- coding: utf-8 -*-
"""Build quant_runs/deploy_table_v4.csv: one homogeneous MAE basis (common first-400 of split 0).

deploy_table_v3.csv inherited its MAE columns from perimage npz dumps of two sizes
(30 runs stored 400 images, 36 stored 800), so its columns are not comparable across runs.
Here every row is recomputed on ONE common 400-image subset, picked by row id.
"""
import csv, hashlib, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3B = os.path.join(ROOT, "quant_runs", "deploy_table_v3b.csv")
V4  = os.path.join(ROOT, "quant_runs", "deploy_table_v4.csv")
PERI = os.path.join(ROOT, "quant_runs", "perimage")
JS  = os.path.join(ROOT, "quant_runs", "deploy_table_v4_report.json")
BITS = ["fp32", "8bit", "6bit", "4bit"]

def npz_path(run, bit):
    return os.path.join(PERI, "%s__%s.npz" % (run, bit))

def load(run, bit):
    p = npz_path(run, bit)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["pred"].astype(np.float64), z["target"].astype(np.float64), z["rows"].astype(np.int64)

ref = load("utk-relu_qat-none-s0-1793011336", "fp32")
COMMON = ref[2][:400]
assert COMMON.size == 400, COMMON.size

rows = list(csv.DictReader(open(V3B, encoding="utf-8")))
out, report = [], {"basis": "common first400 of split_random(seed 0) val", "common_rows_sha1": hashlib.sha1(COMMON.tobytes()).hexdigest(),
                   "n_rows": len(rows), "recomputed": 0, "kept_csv": [], "dropped": []}
for r in rows:
    run = r["run"]
    z = load(run, "fp32")
    if z is None:
        report["kept_csv"].append(run); out.append(dict(r, n_eval="csv")); continue
    pos = {int(rid): i for i, rid in enumerate(z[2])}
    if not all(int(c) in pos for c in COMMON):
        report["dropped"].append(run); continue
    idx = np.array([pos[int(c)] for c in COMMON])
    y = z[1][idx]
    new = dict(r)
    for b in BITS:
        zb = load(run, b)
        if zb is None:
            new["mae_" + b if b != "fp32" else "mae_fp32"] = ""
            continue
        pb = zb[0][[pos[int(c)] for c in COMMON]]
        val = float(np.abs(pb - y).mean())
        new["mae_fp32" if b == "fp32" else "mae_" + b[0]] = "%.4f" % val
    new["n_eval"] = "400"
    report["recomputed"] += 1
    out.append(new)

cols = list(rows[0].keys()) + ["n_eval"]
with open(V4, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in out:
        w.writerow({k: r.get(k, "") for k in cols})

# how much did the 800-image rows move?
moved = 0
maxd = 0.0
by_run = {r["run"]: r for r in rows}
for r in out:
    if r["n_eval"] != "400":
        continue
    o = by_run[r["run"]]
    for k in ["mae_fp32", "mae_8", "mae_6", "mae_4"]:
        if not o[k] or not r[k]:
            continue
        d = abs(float(o[k]) - float(r[k]))
        if d > 1e-9:
            moved += 1
            maxd = max(maxd, d)
report["cells_changed"] = moved
report["max_abs_change"] = round(maxd, 6)
json.dump(report, open(JS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("rows_in=%d rows_out=%d recomputed=%d kept_csv=%d dropped=%d cells_changed=%d max|d|=%.4f"
      % (len(rows), len(out), report["recomputed"], len(report["kept_csv"]), len(report["dropped"]), moved, maxd))
print("kept_csv:", report["kept_csv"][:5], "dropped:", report["dropped"][:5])
