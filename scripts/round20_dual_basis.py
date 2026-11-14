# -*- coding: utf-8 -*-
"""Four run-level contrasts on two evaluation bases: first400 (M=400) and held-out (M=1556)."""
import csv, json, math, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4 = os.path.join(ROOT, "quant_runs", "deploy_table_v4.csv")
HO = os.path.join(ROOT, "quant_runs", "heldout_eval_v3.csv")
OUT = os.path.join(ROOT, "quant_runs", "dual_basis_report.json")
MD = os.path.join(ROOT, "quant_runs", "dual_basis_report.md")

def idx_v4():
    d = {}
    for r in csv.DictReader(open(V4, encoding="utf-8")):
        if r["split_seed"] != "0" or r["ema_decay"] != "0.99":
            continue
        d[(r["mode"], int(r["snap_every"]), int(r["seed"]))] = float(r["mae_fp32"])
    return d

def idx_ho():
    d = {}
    for r in csv.DictReader(open(HO, encoding="utf-8")):
        if r["subset"] != "heldout_1556":
            continue
        d[(r["mode"], int(r["N"]), int(r["seed"]))] = float(r["mae_fp32"])
    return d

A4, HO_ = idx_v4(), idx_ho()
SERIES = {"none_inf": ("none", 0), "none_n1": ("none", 1), "none_n20": ("none", 20),
          "after_n20": ("after_snap", 20), "before_n20": ("before_snap", 20)}
CONTRASTS = [("a", "none_n1", "none_inf"), ("b", "none_n20", "none_inf"),
             ("c", "after_n20", "none_n20"), ("d", "before_n20", "after_n20")]

def series(idx, key):
    m, n = SERIES[key]
    return {s: idx[(m, n, s)] for s in range(8) if (m, n, s) in idx}

def contrast(idx, x, y):
    a, b = series(idx, x), series(idx, y)
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    n = d.size
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tc = float(stats.t.ppf(0.975, n - 1))
    neg = int((d < 0).sum())
    p = float(stats.binomtest(min(neg, n - neg), n, 0.5).pvalue)
    return dict(n=int(n), seeds=seeds, mean=mean, sd=sd, se=se,
                ci=[mean - tc * se, mean + tc * se], neg=neg, sign_p=p,
                per_seed={int(s): float(v) for s, v in zip(seeds, d)})

out = {"bases": {}, "series_missing": {}}
for name, idx in [("first400", A4), ("heldout1556", HO_)]:
    c = {cid: contrast(idx, x, y) for cid, x, y in CONTRASTS}
    out["bases"][name] = c
    out["series_missing"][name] = {k: sorted(set(range(8)) - set(series(idx, k))) for k in SERIES if len(series(idx, k)) < 8}

json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

lines = ["# 双口径 run 层对比（n = 8，df = 7）", "",
         "- first400：deploy_table_v4.csv，M = 400，公共 400 张（split 0）",
         "- heldout1556：heldout_eval_v3.csv 的 subset=heldout，M = 1556，未参与选轮", "",
         "| 编号 | 对比 | first400 均值 [95% CI] | heldout 均值 [95% CI] | 两口径差 |",
         "|---|---|---|---|---|"]
LBL = {"a": "none@N=1 − none@N=∞", "b": "none@N=20 − none@N=∞",
       "c": "after_snap − none @N=20", "d": "before_snap − after_snap @N=20"}
for cid, _, _ in CONTRASTS:
    f, h = out["bases"]["first400"][cid], out["bases"]["heldout1556"][cid]
    lines.append("| %s | %s | %+.6f [%+.6f, %+.6f] | %+.6f [%+.6f, %+.6f] | %+.6f |"
                 % (cid, LBL[cid], f["mean"], f["ci"][0], f["ci"][1],
                    h["mean"], h["ci"][0], h["ci"][1], h["mean"] - f["mean"]))
lines += ["", "| 编号 | first400 负/8 符号 p | heldout 负/8 符号 p |", "|---|---|---|"]
for cid, _, _ in CONTRASTS:
    f, h = out["bases"]["first400"][cid], out["bases"]["heldout1556"][cid]
    lines.append("| %s | %d/8 p=%.4f | %d/8 p=%.4f |" % (cid, f["neg"], f["sign_p"], h["neg"], h["sign_p"]))
open(MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")

for b in ["first400", "heldout1556"]:
    print("==", b)
    for cid, _, _ in CONTRASTS:
        r = out["bases"][b][cid]
        print("  %s n=%d mean=%+.6f ci=[%+.6f,%+.6f] neg=%d p=%.4f" % (cid, r["n"], r["mean"], r["ci"][0], r["ci"][1], r["neg"], r["sign_p"]))
    print("  missing seeds:", out["series_missing"][b])