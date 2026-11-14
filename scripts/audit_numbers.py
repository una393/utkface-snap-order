
import csv, json, statistics as st, sys, glob, os
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:/Users/USER\Desktop\paper-utkface"

rows = list(csv.DictReader(open(os.path.join(ROOT, "quant_runs", "deploy_table.csv"), encoding="utf-8")))
print("cols:", rows[0].keys())
def m(sel, col):
    v = [float(r[col]) for r in rows if sel(r)]
    return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0, len(v))
print("\n-- protocol means per interval (mae_fp32/8/6/4) --")
for N in (1, 5, 20):
    for col in ("mae_fp32", "mae_8", "mae_6", "mae_4"):
        mu, sd, n = m(lambda r, N=N: int(r["snap_every"]) == N, col)
        print(f"N={N:<3}{col:<10} mean={mu:.4f} sd={sd:.4f} n={n}")
print("\n-- 4-bit by interval across all seeds --")
for N in (1, 5, 20):
    v = [float(r["mae_4"]) for r in rows if int(r["snap_every"]) == N]
    print(f"N={N:<3} n={len(v)} mean={st.mean(v):.4f} min={min(v):.4f} max={max(v):.4f} sd={(st.pstdev(v) if len(v)>1 else 0):.4f}")
print("\n-- freq20 seeds: val_mae_ema of before/after --")
for mode in ("before_snap", "after_snap"):
    v = []
    for p in glob.glob(os.path.join(ROOT, "results", "utk-relu_qat-%s-*" % mode, "result.json")):
        d = json.load(open(p, encoding="utf-8"))
        if d.get("snap_every", 1) == 20 and d.get("head", "relu_qat") == "relu_qat":
            v.append(d.get("val_mae_ema"))
    v = [x for x in v if x]
    print(mode, [round(x, 6) for x in v], "mean=%.5f sd=%.5f n=%d" % (st.mean(v), st.pstdev(v), len(v)))

facts = json.load(open(os.path.join(ROOT, "quant_runs", "paper2_facts.json"), encoding="utf-8"))
print("\n-- paper2_facts.json keys --", list(facts)[:10])
def rng(key):
    vals = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == key and isinstance(v, (int, float)):
                    vals.append(float(v))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(facts)
    return (min(vals), max(vals), len(vals)) if vals else None
for k in ("shadow_vs_live", "before_vs_after", "cross_seed", "rel_l2", "residual"):
    r = rng(k)
    if r:
        print(f"{k:<18} min={r[0]:.4f} max={r[1]:.4f} n={r[2]}")
