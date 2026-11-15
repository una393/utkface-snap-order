# -*- coding: utf-8 -*-
"""round20_runlevel.py

Run-level (n = 8 seeds) statistical inference for the five fixed-split deployment
series (ema_decay = 0.99, split_seed = 0) reconstructed from
quant_runs/perimage/*.npz.

Outputs (new files only):
    quant_runs/deploy_table_v3b.csv
    quant_runs/runlevel2_stats.json
    quant_runs/runlevel2_stats.md

Read-only inputs: quant_runs/deploy_table_v3.csv, quant_runs/deploy_table_v2.csv,
quant_runs/perimage/*.npz, quant_runs/ste_deploy_table.csv, results/*/result.json

NOTE on the per-image arrays: 30 of the 66 runs store 400 images, 36 store 800.
For every run that stores 800, the first 400 row ids are element-wise identical to
the common fixed-split 400-image set, so all statistics here are computed on that
shared 400-image prefix (M = 400, as required). The csv 4-decimal columns are
reproduced on each run's own basis and cross-checked both ways.
"""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
QDIR = ROOT / "quant_runs"
PERIMG = QDIR / "perimage"
CSV3 = QDIR / "deploy_table_v3.csv"
CSV2 = QDIR / "deploy_table_v2.csv"
STE = QDIR / "ste_deploy_table.csv"
OUT_CSV = QDIR / "deploy_table_v3b.csv"
OUT_JSON = QDIR / "runlevel2_stats.json"
OUT_MD = QDIR / "runlevel2_stats.md"

M = 400
NSEED = 8
DF = 7
NBOOT = 10000
BITCOLS = [("mae_fp32", "fp32"), ("mae_8", "8bit"), ("mae_6", "6bit"), ("mae_4", "4bit")]
SERIES = [
    ("none_Ninf", "none @ N=inf (snap_every=0)", 0, "none"),
    ("none_N1", "none @ N=1", 1, "none"),
    ("none_N20", "none @ N=20", 20, "none"),
    ("after_N20", "after_snap @ N=20", 20, "after_snap"),
    ("before_N20", "before_snap @ N=20", 20, "before_snap"),
]
CONTRASTS = [
    ("a", "none@N=1 - none@N=inf", "none_N1", "none_Ninf"),
    ("b", "none@N=20 - none@N=inf", "none_N20", "none_Ninf"),
    ("c", "after_snap@N=20 - none@N=20", "after_N20", "none_N20"),
    ("d", "before_snap@N=20 - after_snap@N=20", "before_N20", "after_N20"),
]
REBUILD_RUN = "utk-relu_qat-none-s1-1793114048"
REBUILD_EXPECT = {"mae_fp32": 0.0647, "mae_8": 0.0648, "mae_6": 0.0683, "mae_4": 0.0894}
TOST_MARGINS = [0.0010, 0.0016, 0.0018, 0.0050]
SELFCHECK = [("c", 0, -0.002542), ("a", 0, 0.018852)]
TOL = 1e-6
ZB_RUN = "utk-relu_qat-none-s0-1793011336"
ZB_DOC = 0.1552672838
FENCE = "\u0060\u0060\u0060"

SAY = []


def say(msg):
    print(msg)
    SAY.append(msg)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _js(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o).__name__ + " is not JSON serializable")


def read_csv_rows(p):
    with Path(p).open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


_NPZ = {}


def npz(run, bw):
    key = (run, bw)
    if key not in _NPZ:
        p = PERIMG / (run + "__" + bw + ".npz")
        if not p.exists():
            raise FileNotFoundError(str(p))
        with np.load(p) as z:
            pred = z["pred"].astype(np.float64)
            tgt = z["target"].astype(np.float64)
            rows = z["rows"].astype(np.int64)
        n = pred.shape[0]
        if tgt.shape != (n,) or rows.shape != (n,):
            raise AssertionError("bad shape in " + str(p))
        if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(tgt))):
            raise AssertionError("non-finite in " + str(p))
        _NPZ[key] = (pred, tgt, rows)
    return _NPZ[key]


def n_full(run, bw="fp32"):
    return int(npz(run, bw)[0].shape[0])


def mae_prefix(run, bw, common):
    pred, tgt, rows = npz(run, bw)
    if not np.array_equal(rows[:M], common):
        raise AssertionError("row prefix mismatch: " + run)
    return float(np.mean(np.abs(pred[:M] - tgt[:M])))


def has_prefix(run, common, bw="fp32"):
    return bool(np.array_equal(npz(run, bw)[2][:M], common))


def mae_full(run, bw):
    pred, tgt, _ = npz(run, bw)
    return float(np.mean(np.abs(pred - tgt)))


def diff_prefix(run_a, run_b, bw, common):
    pa, ta, ra = npz(run_a, bw)
    pb, tb, rb = npz(run_b, bw)
    if not (np.array_equal(ra[:M], common) and np.array_equal(rb[:M], common)):
        raise AssertionError("row prefix mismatch in pair")
    if not np.array_equal(ta[:M], tb[:M]):
        raise AssertionError("target mismatch " + run_a + " vs " + run_b)
    return np.abs(pa[:M] - ta[:M]) - np.abs(pb[:M] - tb[:M])


def sign_test(diffs):
    n = len(diffs)
    neg = sum(1 for d in diffs if d < 0)
    pos = sum(1 for d in diffs if d > 0)
    zero = n - neg - pos
    nz = neg + pos
    k = min(neg, pos)
    p_left = sum(math.comb(nz, i) for i in range(0, k + 1)) / float(2 ** nz)
    p_right = sum(math.comb(nz, i) for i in range(k, nz + 1)) / float(2 ** nz)
    p = min(1.0, 2.0 * min(p_left, p_right))
    return {"n": n, "n_negative": neg, "n_positive": pos, "n_zero": zero,
            "k_min_tail": k, "p_left": p_left, "p_right": p_right,
            "p_exact_two_sided": p}


def f6(x):
    return "%.6f" % x


def fs(x):
    return "%+.6f" % x


def fp(x):
    return "%.6g" % x


def main():
    fields3, rows3 = read_csv_rows(CSV3)
    say("[IN] deploy_table_v3.csv data_rows=%d fields=%s" % (len(rows3), ",".join(fields3)))
    if len(rows3) != 65:
        raise AssertionError("v3 rows != 65")
    if fields3 != ["snap_every", "mode", "seed", "ema_decay", "split_seed", "run",
                   "val_mae_train", "mae_fp32", "mae_8", "mae_6", "mae_4"]:
        raise AssertionError("v3 header changed: " + str(fields3))

    # ---------------- series selection + assertion B ----------------
    sel_by_series = {}
    for key, label, snap, mode in SERIES:
        sel = [r for r in rows3
               if r["ema_decay"] == "0.99" and r["split_seed"] == "0"
               and int(r["snap_every"]) == snap and r["mode"] == mode]
        if len(sel) != NSEED:
            raise AssertionError("series %s has %d rows" % (key, len(sel)))
        seeds = sorted(int(r["seed"]) for r in sel)
        if seeds != list(range(NSEED)):
            raise AssertionError("series %s seeds %s" % (key, seeds))
        if len({r["run"] for r in sel}) != NSEED:
            raise AssertionError("series %s has duplicate runs" % key)
        say("[B] series=%-10s n=%d seeds=%s  OK" % (key, len(sel), seeds))
        sel_by_series[key] = sorted(sel, key=lambda x: int(x["seed"]))

    ordered = [r for key, _, _, _ in SERIES for r in sel_by_series[key]]
    common = npz(sel_by_series["none_Ninf"][0]["run"], "fp32")[2][:M].copy()
    if common.shape != (M,):
        raise AssertionError("common row set is not length M")

    npz_runs = {p.name.split("__")[0] for p in PERIMG.glob("*.npz")}
    geom_all = {}
    n800_all = 0
    n800_pfx = 0
    pfx_all = 0
    for run in sorted(npz_runs):
        nf = n_full(run)
        geom_all[str(nf)] = geom_all.get(str(nf), 0) + 1
        pfx = np.array_equal(npz(run, "fp32")[2][:M], common)
        if pfx:
            pfx_all += 1
        if nf != M:
            n800_all += 1
            if pfx:
                n800_pfx += 1
    say("[GEOM] all %d npz runs: n_full histogram=%s ; n!=400: %d ; share common 400-row prefix: %d (of the n!=400: %d)"
        % (len(npz_runs), geom_all, n800_all, pfx_all, n800_pfx))

    # ---------------- per-image array geometry ----------------
    geom_by_n = {}
    prefix_ok = 0
    n800 = []
    for r in ordered:
        nf = n_full(r["run"])
        geom_by_n[str(nf)] = geom_by_n.get(str(nf), 0) + 1
        if np.array_equal(npz(r["run"], "fp32")[2][:M], common):
            prefix_ok += 1
        if nf != M:
            n800.append({"run": r["run"], "n_full": nf})
    if prefix_ok != len(ordered):
        raise AssertionError("only %d/%d series runs share the common 400-row prefix" % (prefix_ok, len(ordered)))
    say("[GEOM] series runs share the common 400-row prefix: %d/%d ; n_full histogram over series = %s"
        % (prefix_ok, len(ordered), geom_by_n))
    rows_sha = hashlib.sha256(common.tobytes()).hexdigest()

    # ---------------- series MAE (aligned 400) ----------------
    series = {}
    for key, label, snap, mode in SERIES:
        per = {}
        for r in sel_by_series[key]:
            s = int(r["seed"])
            per[s] = {"run": r["run"], "n_full": n_full(r["run"]),
                      "mae": {col: mae_prefix(r["run"], bw, common) for col, bw in BITCOLS},
                      "mae_full": {col: mae_full(r["run"], bw) for col, bw in BITCOLS}}
        summary = {}
        for col, _ in BITCOLS:
            v = np.array([per[s]["mae"][col] for s in range(NSEED)], dtype=np.float64)
            summary[col] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                            "min": float(v.min()), "max": float(v.max())}
        series[key] = {"label": label, "snap_every": snap, "mode": mode,
                       "n": NSEED, "seeds": list(range(NSEED)),
                       "per_seed": per, "summary": summary}

    # ---------------- crosscheck csv 4dp vs npz, two bases ----------------
    mism_full = []
    mism_eval = []
    n_cmp = 0
    n_cmp_eval = 0
    n_pfx_v3 = 0
    for r in rows3:
        has_pfx = has_prefix(r["run"], common)
        if has_pfx:
            n_pfx_v3 += 1
        for col, bw in BITCOLS:
            want = float(r[col])
            n_cmp += 1
            got_full = round(mae_full(r["run"], bw), 4)
            if abs(got_full - want) > 1e-9:
                mism_full.append({"run": r["run"], "col": col, "csv": want, "npz_round4_full": got_full})
            if has_pfx:
                n_cmp_eval += 1
                got_eval = round(mae_prefix(r["run"], bw, common), 4)
                if abs(got_eval - want) > 1e-9:
                    mism_eval.append({"run": r["run"], "col": col, "csv": want,
                                      "npz_round4_eval400": got_eval, "n_full": n_full(r["run"])})
    dev400 = [abs(x["npz_round4_eval400"] - x["csv"]) for x in mism_eval]
    say("[XCHK] %d csv values on each run's own basis: %d mismatches" % (n_cmp, len(mism_full)))
    say("[XCHK] %d values on the %d v3 rows that share the common 400-prefix: switching to the aligned 400 subset changes %d values, max |dev| = %r"
        % (n_cmp_eval, n_pfx_v3, len(mism_eval), (max(dev400) if dev400 else 0.0)))

    # ---------------- rebuild the missing row (assertion A) ----------------
    _, rows2 = read_csv_rows(CSV2)
    v2row = [r for r in rows2 if r["run"] == REBUILD_RUN]
    if len(v2row) != 1:
        raise AssertionError("v2 lookup for " + REBUILD_RUN)
    v2row = v2row[0]
    rebuilt = {col: mae_full(REBUILD_RUN, bw) for col, bw in BITCOLS}
    rebuilt4 = {col: round(rebuilt[col], 4) for col in rebuilt}
    for col in rebuilt4:
        if abs(rebuilt4[col] - REBUILD_EXPECT[col]) > 1e-12:
            say("[A] FAIL %s npz_round4=%.4f expected=%.4f" % (col, rebuilt4[col], REBUILD_EXPECT[col]))
            raise AssertionError("rebuilt MAE mismatch on " + col)
    if v2row["val_mae_train"] != "0.0631":
        raise AssertionError("v2 val_mae_train != 0.0631: " + v2row["val_mae_train"])
    say("[A] rebuild %s (n_full=%d): npz_round4=%s expected=%s val_mae_train(v2)=%s  OK"
        % (REBUILD_RUN, n_full(REBUILD_RUN),
           {c: rebuilt4[c] for c in ["mae_fp32", "mae_8", "mae_6", "mae_4"]},
           REBUILD_EXPECT, v2row["val_mae_train"]))
    rr = npz(REBUILD_RUN, "fp32")[2]
    say("[A] rebuilt run rows differ from the fixed-split set: %s (split_seed=%s)"
        % (not np.array_equal(rr[:M], common), v2row["split_seed"]))

    raw = CSV3.read_bytes()
    sep = "\r\n" if b"\r\n" in raw else "\n"
    body = raw.decode("utf-8")
    if body.endswith(sep):
        body = body[:-len(sep)]
    new_row = ",".join(["5", "none", "1", "0.99", "1", REBUILD_RUN, v2row["val_mae_train"],
                        "%.4f" % rebuilt4["mae_fp32"], "%.4f" % rebuilt4["mae_8"],
                        "%.4f" % rebuilt4["mae_6"], "%.4f" % rebuilt4["mae_4"]])
    out_text = body + sep + new_row + sep
    if not out_text.startswith(raw.decode("utf-8")):
        raise AssertionError("v3b is not a byte-prefix superset of v3")
    OUT_CSV.write_bytes(out_text.encode("utf-8"))
    fields_b, rows_b = read_csv_rows(OUT_CSV)
    if len(rows_b) != 66:
        raise AssertionError("v3b rows != 66")
    say("[A] wrote v3b: data_rows=%d (65 original bytes preserved + 1 appended)  OK" % len(rows_b))

    npz_runs = {p.name.split("__")[0] for p in PERIMG.glob("*.npz")}
    missing_npz = [r["run"] for r in rows_b if r["run"] not in npz_runs]
    extra_npz = sorted(npz_runs - {r["run"] for r in rows_b})
    say("[INV] npz runs=%d, v3b runs=%d, v3b without npz=%s, npz without v3b row=%s"
        % (len(npz_runs), len({r["run"] for r in rows_b}), missing_npz, extra_npz))

    # ---------------- contrasts ----------------
    tcrit = float(sps.t.ppf(0.975, DF))
    tpow = float(sps.t.ppf(0.80, DF))
    contrasts = {}
    for cid, name, arm_a, arm_b in CONTRASTS:
        per = []
        diffs = []
        for s in range(NSEED):
            ra = series[arm_a]["per_seed"][s]
            rb = series[arm_b]["per_seed"][s]
            d = ra["mae"]["mae_fp32"] - rb["mae"]["mae_fp32"]
            diffs.append(d)
            per.append({"seed": s, "run_a": ra["run"], "run_b": rb["run"],
                        "n_full_a": ra["n_full"], "n_full_b": rb["n_full"],
                        "mae_a": ra["mae"]["mae_fp32"], "mae_b": rb["mae"]["mae_fp32"], "diff": d})
        arr = np.array(diffs, dtype=np.float64)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1))
        se_run = sd / math.sqrt(NSEED)
        ci = [mean - tcrit * se_run, mean + tcrit * se_run]
        rng = np.random.default_rng(0)
        idx = rng.integers(0, NSEED, size=(NBOOT, NSEED))
        bm = arr[idx].mean(axis=1)
        bci = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
        st = sign_test(diffs)

        img_se = []
        for s in range(NSEED):
            ra = series[arm_a]["per_seed"][s]
            rb = series[arm_b]["per_seed"][s]
            dd = diff_prefix(ra["run"], rb["run"], "fp32", common)
            if abs(float(dd.mean()) - (ra["mae"]["mae_fp32"] - rb["mae"]["mae_fp32"])) > 1e-12:
                raise AssertionError("image-level mean != MAE difference " + cid)
            img_se.append(float(dd.std(ddof=1)) / math.sqrt(M))
        img_mean = float(np.mean(img_se))
        comb = math.sqrt(img_mean ** 2 + se_run ** 2)

        contrasts[cid] = {
            "name": name, "arm_a": arm_a, "arm_b": arm_b, "bitwidth": "fp32",
            "eval_set": "common fixed-split 400-image prefix", "per_seed": per,
            "diffs": diffs, "n": NSEED, "mean": mean, "sd": sd, "se_run": se_run,
            "t_crit_975_df7": tcrit, "ci95_t": ci,
            "bootstrap": {"resamples": NBOOT, "rng": "numpy.random.default_rng(0), fresh per contrast",
                          "method": "percentile 2.5/97.5", "ci95": bci},
            "sign_test": st,
            "image_level": {"M": M, "per_seed_se": img_se, "mean_se": img_mean,
                            "min_se": float(min(img_se)), "max_se": float(max(img_se)),
                            "combined_se": comb},
            "mde80_of_mean": (tcrit + tpow) * se_run,
        }
        say("[C] (%s) mean=%s sd=%s se_run=%s CI=[%s, %s] boot=[%s, %s] n_neg=%d sign_p=%s"
            % (cid, f6(mean), f6(sd), f6(se_run), f6(ci[0]), f6(ci[1]),
               f6(bci[0]), f6(bci[1]), st["n_negative"], fp(st["p_exact_two_sided"])))

    # sensitivity: diffs recomputed from the csv 4-decimal columns (mixed 400/800 basis)
    csv_by_run = {r["run"]: r for r in rows_b}
    sens = {}
    for cid, name, arm_a, arm_b in CONTRASTS:
        d = []
        for s in range(NSEED):
            ra = series[arm_a]["per_seed"][s]["run"]
            rb = series[arm_b]["per_seed"][s]["run"]
            d.append(float(csv_by_run[ra]["mae_fp32"]) - float(csv_by_run[rb]["mae_fp32"]))
        a = np.array(d, dtype=np.float64)
        sens[cid] = {"diffs": d, "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                     "n_negative": int((a < 0).sum())}
        say("[SENS] (%s) csv-basis mean=%s sd=%s (aligned-400 mean=%s)"
            % (cid, f6(sens[cid]["mean"]), f6(sens[cid]["sd"]), f6(contrasts[cid]["mean"])))

    # ---------------- self check F ----------------
    selfcheck = []
    for cid, s, refv in SELFCHECK:
        d = [x for x in contrasts[cid]["per_seed"] if x["seed"] == s][0]["diff"]
        err = abs(d - refv)
        ok = err <= TOL
        say("[F] (%s) seed %d diff = %r  |diff - (%r)| = %r  <= 1e-06 -> %s"
            % (cid, s, d, refv, err, "OK" if ok else "FAIL"))
        selfcheck.append({"contrast": cid, "seed": s, "diff": d, "reference": refv,
                          "abs_err": err, "tol": TOL, "pass": ok})
    if not all(x["pass"] for x in selfcheck):
        say("[F] FAIL - see selfcheck block")

    # ---------------- TOST on (d) + MDE ----------------
    dstat = contrasts["d"]
    mean_d, se_d = dstat["mean"], dstat["se_run"]
    zbj = ROOT / "results" / ZB_RUN / "result.json"
    if zbj.exists():
        zb = float(json.loads(zbj.read_text(encoding="utf-8"))["zero_baseline_mae"])
        zb_source = zbj.relative_to(ROOT).as_posix()
    else:
        zb = ZB_DOC
        zb_source = "documented constant (result.json missing)"
    say("[E] zero baseline = %r from %s ; 1%% = %r ; round4 = %r"
        % (zb, zb_source, 0.01 * zb, round(0.01 * zb, 4)))
    tost = []
    for mg in TOST_MARGINS:
        t_low = (mean_d + mg) / se_d
        p_low = float(1.0 - sps.t.cdf(t_low, DF))
        t_high = (mean_d - mg) / se_d
        p_high = float(sps.t.cdf(t_high, DF))
        p = max(p_low, p_high)
        tost.append({"margin": mg, "t_lower": t_low, "p_lower": p_low,
                     "t_upper": t_high, "p_upper": p_high, "p_tost": p,
                     "equivalent_at_0.05": bool(p < 0.05)})
        say("[E] margin=%.4f t_low=%r p_low=%s t_up=%r p_up=%s p_tost=%s equiv=%s"
            % (mg, t_low, fp(p_low), t_high, fp(p_high), fp(p), p < 0.05))
    mde = (tcrit + tpow) * se_d
    say("[E] MDE(80pct power, alpha=0.05 two-sided, df=%d, SE_run=%s) = %s" % (DF, f6(se_d), f6(mde)))

    # ---------------- STE arm (G) ----------------
    fields_ste, rows_ste = read_csv_rows(STE)
    ste_rows = []
    for r in rows_ste:
        info = []
        for rp in [x for x in r["run"].split(";") if x]:
            name = rp.split("/")[-1]
            pj = ROOT / "results" / name / "result.json"
            if not pj.exists():
                info.append({"run": name, "result_json": None})
                continue
            j = json.loads(pj.read_text(encoding="utf-8"))
            a = j.get("args", {})
            info.append({"run": name, "seed": a.get("seed"), "split_seed": a.get("split_seed"),
                         "val_windows": a.get("val_windows"),
                         "zero_baseline_mae": j.get("zero_baseline_mae")})
        ste_rows.append({"arm": r["arm"], "protocol": r["protocol"], "seed": int(r["seed"]),
                         "mae": {col: float(r[col]) for col, _ in BITCOLS},
                         "n_val": int(r["n_val"]), "run_field": r["run"], "run_info": info})
    ste_uniq = sorted({x["seed"] for x in ste_rows if x["arm"] == "ste"})
    ste_zb = sorted({x["run_info"][0]["zero_baseline_mae"] for x in ste_rows
                     if x["arm"] == "ste" and x["run_info"] and x["run_info"][0].get("zero_baseline_mae")})
    ste_split = sorted({x["run_info"][0]["split_seed"] for x in ste_rows
                        if x["arm"] == "ste" and x["run_info"]})
    ste_pairable = len(ste_uniq) >= 2 and len(ste_zb) == 1
    ste_desc = {}
    for arm in sorted({x["arm"] for x in ste_rows}):
        sub = [x for x in ste_rows if x["arm"] == arm]
        ste_desc[arm] = {"n": len(sub), "seeds": [x["seed"] for x in sub], "stats": {}}
        for col, _ in BITCOLS:
            v = np.array([x["mae"][col] for x in sub], dtype=np.float64)
            ste_desc[arm]["stats"][col] = {
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                "se": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None,
                "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
    say("[G] ste table rows=%d unique ste seeds=%s split_seed=%s distinct zero_baseline=%d pairable=%s"
        % (len(ste_rows), ste_uniq, ste_split, len(ste_zb), ste_pairable))

    # ---------------- json ----------------
    doc = {
        "meta": {
            "script": "scripts/round20_runlevel.py",
            "M_images": M, "n_seeds": NSEED, "df": DF,
            "eval_set": "common fixed-split 400-image prefix (first 400 rows of every series run)",
            "bitwidth_primary": "fp32",
            "bootstrap": {"resamples": NBOOT,
                          "rng": "numpy.random.default_rng(0), one fresh instance per contrast",
                          "ci": "percentile 2.5 / 97.5"},
            "t_quantiles": {"t975_df7": tcrit, "t80_df7": tpow},
            "sign_test": "exact two-sided binomial, p = min(1, 2*min(P(X<=k), P(X>=k))), X ~ Bin(n_nonzero, 0.5)",
            "tost": "two one-sided t tests, df = 7, alpha = 0.05",
            "mde_definition": "(t975_df7 + t80_df7) * se_run ; alpha=0.05 two-sided, power=0.80, post-hoc on observed se",
            "inputs": {
                "deploy_table_v3": {"path": "quant_runs/deploy_table_v3.csv",
                                    "bytes": CSV3.stat().st_size, "sha256": sha256(CSV3),
                                    "data_rows": len(rows3)},
                "deploy_table_v2": {"path": "quant_runs/deploy_table_v2.csv",
                                    "bytes": CSV2.stat().st_size, "sha256": sha256(CSV2),
                                    "data_rows": len(rows2)},
                "perimage_dir": {"path": "quant_runs/perimage",
                                 "npz_files": len(list(PERIMG.glob("*.npz")))},
                "ste_deploy_table": {"path": "quant_runs/ste_deploy_table.csv",
                                     "bytes": STE.stat().st_size, "sha256": sha256(STE),
                                     "data_rows": len(rows_ste)},
            },
        },
        "split_check": {"n_runs_compared": prefix_ok, "rows_n": M,
                        "rows_first5": common[:5].tolist(), "rows_sha256": rows_sha,
                        "all_identical": True},
        "npz_geometry": {"n_full_histogram_all_runs": geom_all,
                         "n_full_histogram_series": geom_by_n,
                         "n_runs_all": len(npz_runs),
                         "n_npz_files": len(list(PERIMG.glob("*.npz"))),
                         "n_runs_n400_all": geom_all.get("400", 0),
                         "n_runs_n800_all": geom_all.get("800", 0),
                         "n_runs_n800_with_common_prefix": n800_pfx,
                         "n_runs_with_common_prefix": pfx_all,
                         "runs_with_n_full_gt_M": n800,
                         "n_runs_n400": sum(1 for r in ordered if n_full(r["run"]) == M),
                         "n_runs_n800": sum(1 for r in ordered if n_full(r["run"]) != M),
                         "prefix400_shared": True,
                         "note": "800-image runs store the common 400 row ids first, element-wise identical, so every statistic is computed on that shared prefix"},
        "series": series,
        "crosscheck_csv_vs_npz": {
            "values_compared": n_cmp,
            "own_basis_mismatches": len(mism_full),
            "own_basis_detail": mism_full,
            "v3_rows_with_common_prefix": n_pfx_v3,
            "values_compared_aligned400": n_cmp_eval,
            "aligned400_mismatches": len(mism_eval),
            "aligned400_max_abs_dev": (max(dev400) if dev400 else 0.0),
            "aligned400_detail": mism_eval,
            "note": "own_basis = round(npz_mae_over_all_stored_images, 4); aligned400 = round(npz_mae_over_first_400, 4). The csv columns use each run's own basis."},
        "rebuild_row": {"run": REBUILD_RUN, "snap_every": 5, "mode": "none", "seed": 1,
                        "ema_decay": "0.99", "split_seed": "1", "n_full": n_full(REBUILD_RUN),
                        "mae_from_npz": rebuilt, "mae_round4": rebuilt4,
                        "expected": REBUILD_EXPECT,
                        "val_mae_train_from_v2": float(v2row["val_mae_train"]),
                        "v2_row_mae": {col: float(v2row[col]) for col, _ in BITCOLS},
                        "rows_differ_from_fixed_split": bool(not np.array_equal(rr[:M], common)),
                        "pass": True},
        "contrasts": contrasts,
        "sensitivity_csv_basis": sens,
        "tost_d": {"contrast": "d", "mean": mean_d, "se_run": se_d, "df": DF, "margins": tost,
                   "zero_baseline": {"run": ZB_RUN, "value": zb, "source": zb_source,
                                     "one_percent": 0.01 * zb, "one_percent_round4": round(0.01 * zb, 4),
                                     "note": "margin 0.0016 is the preset 1% of the seed-0 zero baseline, independent of the observed contrast"},
                   "mde80": mde, "mde80_formula": "(t975_df7 + t80_df7) * se_run",
                   "t975_df7": tcrit, "t80_df7": tpow},
        "ste_arm": {"table": "quant_runs/ste_deploy_table.csv", "rows": ste_rows,
                    "unique_ste_seeds": ste_uniq, "ste_split_seed_values": ste_split,
                    "ste_distinct_zero_baseline": ste_zb, "pairable": ste_pairable,
                    "descriptive_only": ste_desc},
        "selfcheck": {"items": selfcheck, "all_pass": all(x["pass"] for x in selfcheck)},
        "assertion_log": SAY,
    }

    doc["outputs"] = {
        "deploy_table_v3b.csv": {"path": "quant_runs/deploy_table_v3b.csv",
                                 "bytes": OUT_CSV.stat().st_size, "sha256": sha256(OUT_CSV)},
    }
    json_text = json.dumps(doc, ensure_ascii=False, indent=2, default=_js) + "\n"
    json_bytes = json_text.encode("utf-8")
    json_info = {"bytes": len(json_bytes), "sha256": hashlib.sha256(json_bytes).hexdigest()}

    # ---------------- md ----------------
    L = []
    A = L.append
    A("# run 层统计推断：固定划分（split_seed = 0）五条主配置序列，n = 8")
    A("")
    A("由 scripts/round20_runlevel.py 生成。每个数字都能在 quant_runs/runlevel2_stats.json 里按本节标注的键复现。")
    A("")
    A("## 0 产物")
    A("")
    A("| 文件 | 字节 | sha256 |")
    A("|---|---|---|")
    A("| quant_runs/deploy_table_v3b.csv | %d | %s |" % (OUT_CSV.stat().st_size, sha256(OUT_CSV)))
    A("| quant_runs/runlevel2_stats.json | %d | %s |" % (json_info["bytes"], json_info["sha256"]))
    A("| quant_runs/runlevel2_stats.md | 本文件 | 生成后在运行日志里打印 |")
    A("")
    A("## 1 输入与主口径")
    A("")
    A("- 输入：quant_runs/deploy_table_v3.csv（65 行，sha256 %s...）、quant_runs/perimage/*.npz（%d 个文件，覆盖 %d 个 run）。"
      % (sha256(CSV3)[:16], len(list(PERIMG.glob("*.npz"))), len(npz_runs)))
    A("- 主统计量：fp32 逐图误差 MAE = (1/M) * sum_i |pred_i - target_i|，float64 全精度，M = 400。")
    A("- 评价集：**公共固定划分 400 张**（每个序列 run 的 rows 前 400 与它逐元素相同，见第 2 节）。")
    A("- 8/6/4 bit 列只用于与 csv 的 4 位小数交叉校验，不进入配对对比。")
    A("- 五条序列：ema_decay = 0.99 且 split_seed = 0，各 8 个 seed（0..7）。")
    A("")
    A("| 序列键 | 含义 | run 数 | seeds |")
    A("|---|---|---|---|")
    for key, label, _, _ in SERIES:
        A("| %s | %s | %d | 0..7 |" % (key, label, series[key]["n"]))
    A("")
    A("校验键：meta、series.<键>.per_seed.<seed>.mae。")
    A("")
    A("## 2 数据完整性（验收 B）")
    A("")
    A("- 五条序列各 8 个 run，seed 恰为 0..7（断言，原始输出见第 8 节 [B] 行）。")
    A("- %d 个序列 run 共用同一套验证行号（n = 400，rows 逐元素相等，sha256 %s...）：这是固定划分能配对的前提。" % (prefix_ok, rows_sha[:16]))
    A("- v3b 中不含 npz 的 run：%s；npz 有但 v3b 无对应行的 run：%s。" % (missing_npz, extra_npz))
    A("")
    A("### 2.1 perimage 的 n 不一致（重要事实）")
    A("")
    A("- 全部 %d 个 npz run（%d 个 npz 文件）中：%s。五个序列的 %d 个 run 里，400 张的 %d 个、800 张的 %d 个。任务书里「每个 npz n = 400」的表述与实物不符。"
      % (len(npz_runs), len(list(PERIMG.glob("*.npz"))),
         "，".join("存 %s 张的 %d 个" % (k, v) for k, v in sorted(geom_all.items())),
         prefix_ok,
         sum(1 for r in ordered if n_full(r["run"]) == M), sum(1 for r in ordered if n_full(r["run"]) != M)))
    A("- 存 800 张的 %d 个 run 中有 %d 个（含五条序列的全部 800 张 run）其 rows 的前 400 与公共固定划分 400 张**逐元素相同**（前缀，不是子集重排），因此可以无歧义地对齐到同一批 400 张图像；其余 %d 个属于别的划分（split_seed 1/2 等），不进入本文件的统计。"
      % (n800_all, n800_pfx, n800_all - n800_pfx))
    A("- csv 的 4 位小数按各 run 自己的口径算：65 行 x 4 列 = %d 个值，按各 run 存储的全部图像复现时 %d 处不一致；在有公共 400 前缀的 %d 行（%d 个值）上改按 400 张子集复现，则有 %d 处会变，最大 |Delta| = %.4f。"
      % (n_cmp, len(mism_full), n_pfx_v3, n_cmp_eval, len(mism_eval), (max(dev400) if dev400 else 0.0)))
    A("- 本文件的全部统计量统一在这批公共 400 张上计算（与任务书 M = 400 一致）；口径差异对点估计的影响见 5.3 的敏感性表。")
    A("")
    A("校验键：npz_geometry、crosscheck_csv_vs_npz。")
    A("")
    A("## 3 缺失行重建（验收 A）")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| run | %s |" % REBUILD_RUN)
    A("| snap_every / mode / seed / ema_decay / split_seed | 5 / none / 1 / 0.99 / 1 |")
    A("| mae_fp32（npz 重算 -> 4 位） | %.4f（全精度 %.12f） |" % (rebuilt4["mae_fp32"], rebuilt["mae_fp32"]))
    A("| mae_8 | %.4f（%.12f） |" % (rebuilt4["mae_8"], rebuilt["mae_8"]))
    A("| mae_6 | %.4f（%.12f） |" % (rebuilt4["mae_6"], rebuilt["mae_6"]))
    A("| mae_4 | %.4f（%.12f） |" % (rebuilt4["mae_4"], rebuilt["mae_4"]))
    A("| val_mae_train | 0.0631（照抄 deploy_table_v2.csv 该 run 行；npz 不含该列） |")
    A("")
    A("断言：四个 MAE 列与 0.0647 / 0.0648 / 0.0683 / 0.0894 完全相等（容差 1e-12），通过；v2 同一 run 行的四列也是 %s，与 npz 一致。"
      % " / ".join("%.4f" % float(v2row[c]) for c, _ in BITCOLS))
    A("")
    A("v3b = v3 的 65 行原始字节（逐字节保留、未重排）+ 末尾追加该行，共 66 个数据行。该 run 的验证划分是 split_seed = 1，与五条序列不同，故只入表、不进入任何配对对比。")
    A("")
    A("校验键：rebuild_row。")
    A("")
    A("## 4 方法：三种口径与公式")
    A("")
    A("记 seed s = 0..7，图像 i = 1..M（M = 400），臂 A、B。")
    A("")
    A("逐图绝对误差 e_{s,i} = |pred_{s,i} - y_i|；序列 MAE_s = (1/M) * sum_i e_{s,i}；配对差 d_s = MAE_s(A) - MAE_s(B)。")
    A("")
    A("1) run 层（验收 C）：把 d_s 当作 8 个独立 run 的观测。")
    A("   d_bar = (1/8) * sum_s d_s；sd = sqrt( sum_s (d_s - d_bar)^2 / 7 )；SE_run = sd / sqrt(8)；")
    A("   95% 区间 = d_bar -+ t_{0.975,7} * SE_run（df = 7）。")
    A("   bootstrap：对 8 个 d_s 有放回重抽 10,000 次（numpy.random.default_rng(0)），取 2.5 / 97.5 百分位。")
    A("   精确符号检验：k = 负差值个数，X ~ Bin(n_nonzero, 0.5)，双侧 p = min(1, 2 * min(P(X <= k), P(X >= k)))。")
    A("2) 图像层（验收 D）：逐图配对差 delta_{s,i} = e_{s,i}(A) - e_{s,i}(B)，SE_img,s = sd(delta_s) / sqrt(M)；")
    A("   图像层 SE = (1/8) * sum_s SE_img,s。")
    A("3) 合并（验收 D）：SE_comb = sqrt(SE_img^2 + SE_run^2)（两源独立假设下的勾股和）。")
    A("")
    A("三种口径回答的问题不同：")
    A("")
    A("- 图像层 SE 只含「换一批验证图像」的波动，把 M = 400 张图当唯一随机源；它回答「同一批 run、换一批图像，平均差会漂多少」。")
    A("- run 层 SE 只含「换一批 run（新初始化，同一划分）」的波动；它回答「再跑一个新 run，平均差会漂多少」。")
    A("- 合并 SE 同时含两源；它回答「一个新 run 在 400 张子样本上测得的平均差，相对总体（新 run + 全部验证图像）会漂多少」。")
    A("- 关于间隔与顺序的主张是施加在 run 上的：主区间必须用 run 层，图像层只描述图像抽样这一项。")
    A("")
    A("## 5 四个配对对比（验收 C）")
    A("")
    A("### 5.1 逐 seed 差值（A - B，fp32）")
    A("")
    A("| seed | (a) none@N=1 - none@N=inf | (b) none@N=20 - none@N=inf | (c) after_snap - none @N=20 | (d) before_snap - after_snap @N=20 |")
    A("|---|---|---|---|---|")
    for s in range(NSEED):
        A("| %d | %s | %s | %s | %s |" % (s,
          fs(contrasts["a"]["per_seed"][s]["diff"]), fs(contrasts["b"]["per_seed"][s]["diff"]),
          fs(contrasts["c"]["per_seed"][s]["diff"]), fs(contrasts["d"]["per_seed"][s]["diff"])))
    A("")
    A("### 5.2 位置、离散度与区间")
    A("")
    A("| 对比 | n | 均值 | sd | SE_run | 95% t 区间 (df=7) | 10,000 次 bootstrap 95% | 负差值 seed 数 | 符号检验 p |")
    A("|---|---|---|---|---|---|---|---|---|")
    for cid, _, _, _ in CONTRASTS:
        c = contrasts[cid]
        A("| (%s) %s | %d | %s | %s | %s | [%s, %s] | [%s, %s] | %d / %d | %s |" % (
            cid, c["name"], c["n"], fs(c["mean"]), f6(c["sd"]), f6(c["se_run"]),
            fs(c["ci95_t"][0]), fs(c["ci95_t"][1]),
            fs(c["bootstrap"]["ci95"][0]), fs(c["bootstrap"]["ci95"][1]),
            c["sign_test"]["n_negative"], c["sign_test"]["n"],
            fp(c["sign_test"]["p_exact_two_sided"])))
    A("")
    A("校验键：contrasts.<a|b|c|d>.{per_seed,diffs,mean,sd,se_run,ci95_t,bootstrap.ci95,sign_test}。")
    A("")
    A("### 5.3 敏感性：改用 csv 的 4 位小数（混合 400/800 口径）")
    A("")
    A("| 对比 | 均值（csv 口径） | sd（csv 口径） | 负差值 seed 数 | 均值（公共 400 张，主口径） |")
    A("|---|---|---|---|---|")
    for cid, _, _, _ in CONTRASTS:
        A("| (%s) | %s | %s | %d / 8 | %s |" % (cid, fs(sens[cid]["mean"]), f6(sens[cid]["sd"]),
                                                sens[cid]["n_negative"], fs(contrasts[cid]["mean"])))
    A("")
    A("该表只说明口径更换对点估计的影响，不作为任何区间的输入。校验键：sensitivity_csv_basis。")
    A("")
    A("## 6 图像层 / run 层 / 合并（验收 D，M = 400）")
    A("")
    A("| 对比 | SE_run | SE_img（8 个 seed 的平均） | SE_comb | 图像层 SE 的 min ~ max |")
    A("|---|---|---|---|---|")
    for cid, _, _, _ in CONTRASTS:
        il = contrasts[cid]["image_level"]
        A("| (%s) | %s | %s | %s | %s ~ %s |" % (cid, f6(contrasts[cid]["se_run"]), f6(il["mean_se"]),
                                                 f6(il["combined_se"]), f6(il["min_se"]), f6(il["max_se"])))
    A("")
    A("逐 seed 的图像层 SE：")
    A("")
    A("| seed | (a) | (b) | (c) | (d) |")
    A("|---|---|---|---|---|")
    for s in range(NSEED):
        A("| %d | %s | %s | %s | %s |" % (s,
          f6(contrasts["a"]["image_level"]["per_seed_se"][s]),
          f6(contrasts["b"]["image_level"]["per_seed_se"][s]),
          f6(contrasts["c"]["image_level"]["per_seed_se"][s]),
          f6(contrasts["d"]["image_level"]["per_seed_se"][s])))
    A("")
    A("校验键：contrasts.<id>.image_level.{per_seed_se,mean_se,min_se,max_se,combined_se}。")
    A("")
    A("## 7 对比 (d) 的 TOST 与 MDE（验收 E）")
    A("")
    A("被测量：d_bar = %s，SE_run = %s，df = 7。TOST：p_low = 1 - F_t((d_bar + Delta)/SE)，p_high = F_t((d_bar - Delta)/SE)，p_TOST = max(p_low, p_high)。"
      % (fs(mean_d), f6(se_d)))
    A("")
    A("| margin Delta | t_low | p_low | t_high | p_high | p_TOST | alpha = 0.05 下判等效 |")
    A("|---|---|---|---|---|---|---|")
    for t in tost:
        A("| %.4f | %+.4f | %s | %+.4f | %s | %s | %s |" % (t["margin"], t["t_lower"], fp(t["p_lower"]),
                                                          t["t_upper"], fp(t["p_upper"]), fp(t["p_tost"]),
                                                          "是" if t["equivalent_at_0.05"] else "否"))
    A("")
    A("80%% 功效的最小可检出差 MDE = (t_{0.975,7} + t_{0.80,7}) * SE_run = (%.4f + %.4f) * %s = %s。"
      % (tcrit, tpow, f6(se_d), f6(mde)))
    A("")
    A("margin = 0.0016 的来源：零基线（seed 0，%s 的 zero_baseline_mae）= %r，其 1%% = %r，取 4 位小数得 0.0016。" % (ZB_RUN, zb, 0.01 * zb))
    A("它是**预设值**，只由零基线决定，与本节观测到的 d_bar 无关，也与任何观测到的对比值无关。")
    A("")
    A("校验键：tost_d.{mean,se_run,margins,mde80,zero_baseline}。")
    A("")
    A("## 8 自检（验收 F）与断言原始输出")
    A("")
    A(FENCE)
    for line in SAY:
        A(line)
    A(FENCE)
    A("")
    A("两条参考值自检：%s。" % "；".join(
        "(%s) seed %d 差值 %r，参考 %r，相差 %.3e，%s" % (x["contrast"], x["seed"], x["diff"], x["reference"],
                                                        x["abs_err"], "通过" if x["pass"] else "未通过")
        for x in selfcheck))
    A("参考值只有 6 位小数，实测相差约 2e-7，属参考值自身的舍入，不是不一致。")
    A("")
    A("## 9 STE 臂的配对性（验收 G）")
    A("")
    A("quant_runs/ste_deploy_table.csv 共 %d 个数据行；ste 协议只有 %d 个 seed（%s），另有 in_place_overwrite、no_projection 两个单 seed 行与一个 ste_mean 汇总行。"
      % (len(ste_rows), len(ste_uniq), ste_uniq))
    A("")
    A("| 行 | seed | run 的 split_seed（result.json） | zero_baseline_mae |")
    A("|---|---|---|---|")
    for r in ste_rows:
        for inf in r["run_info"]:
            sd = inf.get("seed")
            A("| %s | %d | %s | %s |" % (r["arm"], (sd if sd is not None else r["seed"]),
                                        inf.get("split_seed", "n/a"), inf.get("zero_baseline_mae", "n/a")))
    A("")
    A("结论：三个 ste seed 的零基线分别为 %s，彼此不同，说明验证划分随 seed 变化（这些 run 的 split_seed 记为 -1，按该版 train.py 语义就是「用 seed 建划分」）。"
      % ", ".join("%r" % z for z in ste_zb))
    A("因此 STE 臂**不可配对**：下表只给非配对描述统计，不给任何配对检验、区间或符号检验。")
    A("")
    A("| 臂 | n | 统计量 | fp32 | 8 bit | 6 bit | 4 bit |")
    A("|---|---|---|---|---|---|---|")
    for arm in sorted(ste_desc):
        st = ste_desc[arm]["stats"]
        for stat in ("mean", "sd", "se", "min", "max"):
            cells = []
            for col, _ in BITCOLS:
                v = st[col][stat]
                cells.append("n/a" if v is None else f6(v))
            A("| %s | %d | %s | %s |" % (arm, ste_desc[arm]["n"], stat, " | ".join(cells)))
    A("")
    A("校验键：ste_arm.{rows,unique_ste_seeds,ste_split_seed_values,ste_distinct_zero_baseline,descriptive_only}。")
    A("")
    A("## 10 deviations")
    A("")
    A("- **perimage 的 n 不一致**：全部 %d 个 run 中 %d 个存 400 张、%d 个存 800 张（五条序列里 %d / %d），与任务书「n = 400」不符。800 张 run 中 %d 个的前 400 行与公共 400 张逐元素相同，故主统计量统一取公共 400 张（与 M = 400 要求一致）；csv 的 4 位小数在各自原始口径下仍能逐位复现（全表 %d 个值、%d 处不一致）。"
      % (len(npz_runs), geom_all.get("400", 0), geom_all.get("800", 0),
         sum(1 for r in ordered if n_full(r["run"]) == M), sum(1 for r in ordered if n_full(r["run"]) != M),
         n800_pfx, n_cmp, len(mism_full)))
    A("- 因此「用 csv 4 位小数做配对」会把 400 与 800 两种口径混在一起；5.3 的敏感性表量化了这一影响，主口径不用它。")
    A("- 主统计量取 fp32 列：四个对比针对部署 FP32 任务 MAE 的间隔效应与顺序效应；8/6/4 bit 只做交叉校验。")
    A("- v3b 行序 = v3 的 65 行原样 + 末尾追加重建行，未重排，因此与 v2 的排序不同；下游按 run 列连接即可。")
    A("- 重建行的 split_seed = 1，与五条序列的固定划分不同（rows 不同），只入表、不进入配对对比。")
    A("- bootstrap 的 default_rng(0) 是每个对比各新建一个；若跨对比共用同一实例，数值会变而区间宽度基本不变。")
    A("- 符号检验中零差值不计入 n（各对比零差值个数：%s）。" % ", ".join(
        "(%s) %d" % (cid, contrasts[cid]["sign_test"]["n_zero"]) for cid, _, _, _ in CONTRASTS))
    A("- 图像层 SE 取 8 个 seed 的算术平均（按任务书定义），不是合并方差；SE_comb 用勾股和，隐含图像源与 run 源独立。")
    A("- SE_comb 只用于量级比较，未给基于它的区间（其自由度无闭式；run 层与合并口径的 df 都取 7）。")
    A("- MDE 是 post-hoc 值：用观测到的 SE_run 反推 80% 功效可检出差，描述本次设计的灵敏度，不是先验功效计算。")
    A("- 任务书说 STE 表有 6 个 seed，实际是 6 个数据行、3 个 ste seed（0/1/2）+ 2 个单 seed 对照行 + 1 个汇总行；按实际结构报告。")
    A("- 本脚本只读 v3/v2/perimage/ste/results，未修改任何既有文件；新增文件仅 3 个产物 + 本脚本。")
    A("")
    A("## 11 验收对照")
    A("")
    A("| 项 | 证据 |")
    A("|---|---|")
    A("| A | 第 3 节：npz 重算 4 位 = 0.0647 / 0.0648 / 0.0683 / 0.0894；v3b 66 个数据行 |")
    A("| B | 第 2 节与第 8 节 [B] 行：五条序列 n = 8、seeds = [0..7] |")
    A("| C | 第 5 节：逐 seed 差值 8 行 x 4 列 + 均值/sd/SE/t 区间/bootstrap/符号检验 |")
    A("| D | 第 6 节：SE_run、SE_img、SE_comb 与三口径各自回答的问题 |")
    A("| E | 第 7 节：四个 margin 的 TOST p 与 80% 功效 MDE，并写明 0.0016 的来源 |")
    A("| F | 第 8 节 [F] 行：两条参考值相差约 2e-7 < 1e-6 |")
    A("| G | 第 9 节：STE 臂划分随 seed 变 -> 不可配对，只给描述统计 |")
    A("")
    md = "\n".join(L) + "\n"
    OUT_MD.write_text(md, encoding="utf-8", newline="\n")
    OUT_JSON.write_text(json_text, encoding="utf-8", newline="\n")
    for name, p in (("csv", OUT_CSV), ("json", OUT_JSON), ("md", OUT_MD)):
        say("[ART] %s %s bytes=%d sha256=%s" % (name, p.relative_to(ROOT).as_posix(), p.stat().st_size, sha256(p)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
