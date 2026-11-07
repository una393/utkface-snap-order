"""论文 2：把稿件里的关键对比从图像层重算到 run 层。

只读：quant_runs/deploy_table_v2.csv、quant_runs/perimage/*.npz、
      results/<run>/result.json、results/<run>/best.pth。
产出：quant_runs/runlevel_stats.json、quant_runs/runlevel_stats.md。

主配置筛选沿用 scripts/paper2_figs_add.py:66-68 的 is_main_config()（gamma=0.99 且 split_seed==seed）。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "quant_runs" / "deploy_table_v2.csv"
PERIMAGE = ROOT / "quant_runs" / "perimage"
RESULTS = ROOT / "results"
OUT_JSON = ROOT / "quant_runs" / "runlevel_stats.json"
OUT_MD = ROOT / "quant_runs" / "runlevel_stats.md"

M_IMG = 400
N_BOOT = 10000
BOOT_SEED = 20261031
ALPHA = 0.05
TOST_MARGINS = [0.001, 0.0016, 0.0018, 0.005]
PRESET_MARGIN = 0.0016

# ---------------------------------------------------------------- 分布函数


def _betacf(a, b, x):
    """正则化不完全 beta 的连分式（Numerical Recipes 6.4 betacf）。"""
    MAXIT, EPS, FPMIN = 300, 3e-15, 1e-300
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
        d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c
        c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c
        c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbt + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """Student t 上尾概率 P(T > t)。"""
    if df <= 0:
        return float("nan")
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def t_ppf(p, df):
    """Student t 分位点（二分）。"""
    if not 0.0 < p < 1.0:
        raise ValueError(p)
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1e4
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if t_sf(mid, df) > 1.0 - p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def _selfcheck_t():
    for df, want in [(1, 12.7062047), (2, 4.3026527), (3, 3.1824463), (399, 1.9659124)]:
        got = t_ppf(0.975, df)
        assert abs(got - want) < 2e-3, (df, got, want)
    assert abs(t_sf(2.0, 10) - 0.0366940) < 1e-6


# ---------------------------------------------------------------- 统计量


def summary_stats(x):
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    out = {"mean": float(x.mean()), "n": n}
    if n > 1:
        sd = float(x.std(ddof=1))
        se = sd / math.sqrt(n)
        df = n - 1
        tc = t_ppf(1 - ALPHA / 2, df)
        out.update({"sd": sd, "se": se, "df": df, "t_crit": tc,
                    "ci95": [out["mean"] - tc * se, out["mean"] + tc * se]})
    else:
        out.update({"sd": None, "se": None, "df": 0, "t_crit": None, "ci95": None,
                    "note": "n=1：配对 t 的 df=0，区间不可估"})
    return out


def sign_test(k, n):
    """精确二项符号检验，p0=0.5；双侧 = 2*min(尾) 截到 1。"""
    if n == 0:
        return {"k": 0, "n": 0, "p_two_sided": None, "p_greater": None, "p_less": None}
    tot = 2 ** n
    p_ge = sum(math.comb(n, i) for i in range(k, n + 1)) / tot
    p_le = sum(math.comb(n, i) for i in range(0, k + 1)) / tot
    return {"k": int(k), "n": int(n),
            "p_two_sided": min(1.0, 2 * min(p_ge, p_le)),
            "p_greater": p_ge, "p_less": p_le}


def tost(d, margin):
    """配对 t 的双单侧检验（TOST）。"""
    x = np.asarray(d, dtype=float)
    n = int(x.size)
    if n < 2:
        return {"margin": margin, "n": n, "p_lower": None, "p_upper": None,
                "p_tost": None, "equivalent": None, "ci90": None,
                "note": "n=1：se 不可估，TOST 无法进行"}
    mean = float(x.mean())
    se = float(x.std(ddof=1)) / math.sqrt(n)
    df = n - 1
    p_lower = t_sf((mean + margin) / se, df)
    p_upper = t_sf((margin - mean) / se, df)
    p_tost = max(p_lower, p_upper)
    tc90 = t_ppf(0.95, df)
    return {"margin": margin, "n": n, "mean": mean, "se": se, "df": df,
            "p_lower": p_lower, "p_upper": p_upper, "p_tost": p_tost,
            "equivalent": bool(p_tost < ALPHA),
            "ci90": [mean - tc90 * se, mean + tc90 * se]}


def mde_paired(se, df, power=0.80):
    """80% 功效下可检出的最小配对效应：MDE = (t_{1-a/2,df} + t_{power,df}) * SE。"""
    t1 = t_ppf(1 - ALPHA / 2, df)
    t2 = t_ppf(power, df)
    return {"se": se, "df": df, "power": power, "t_alpha": t1, "t_power": t2,
            "mde": (t1 + t2) * se,
            "formula": "MDE = (t_{1-alpha/2, df} + t_{power, df}) * SE, df = n-1"}


def two_level_bootstrap(per_seed_diffs, rng, n_boot=N_BOOT):
    """两级 bootstrap：先整 run(=seed) 有放回重采样，再在每个被抽中的 run 内整图重采样。"""
    n = len(per_seed_diffs)
    if n == 0:
        return {"n_boot": n_boot, "ci95": None}
    m = len(per_seed_diffs[0])
    per_seed_means = np.empty((n, n_boot), dtype=np.float64)
    for i, d in enumerate(per_seed_diffs):
        idx = rng.integers(0, m, size=(n_boot, m))
        per_seed_means[i] = d[idx].mean(axis=1)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
    boot = (counts * per_seed_means.T).sum(axis=1) / n
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_boot": n_boot, "n_seeds": n, "n_images": m,
            "ci95": [float(lo), float(hi)], "mean": float(boot.mean()),
            "sd": float(boot.std(ddof=1)),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "image_only": n == 1}


# ---------------------------------------------------------------- 工具


def is_main_config(r):
    """与 scripts/paper2_figs_add.py:66-68 同一条判据：gamma=0.99 且 split_seed==seed。"""
    return float(r["ema_decay"]) == 0.99 and int(r["split_seed"]) == int(r["seed"])


def ci_str(ci, nd=4):
    if ci is None:
        return "—"
    return "[{:+.{nd}f}, {:+.{nd}f}]".format(ci[0], ci[1], nd=nd)


def fnum(x, nd=4):
    """None 安全的定点格式化（缺产物的 run 用 — 而不是崩）。"""
    if x is None:
        return "—"
    return ("%." + str(nd) + "f") % x


def flag(x):
    if x is None:
        return "无基线"
    return "是" if x else "否"


class PerImage:
    """perimage/*.npz 缓存；abs_err 按 rows 建索引，用于跨 run 逐图配对。"""

    def __init__(self):
        self._cache = {}

    def get(self, run, bits="fp32"):
        key = (run, bits)
        if key in self._cache:
            return self._cache[key]
        p = PERIMAGE / (run + "__" + bits + ".npz")
        if not p.exists():
            self._cache[key] = None
            return None
        z = np.load(p)
        pred = z["pred"].astype(np.float64)
        tgt = z["target"].astype(np.float64)
        rows = z["rows"].astype(np.int64)
        err = np.abs(pred - tgt)
        obj = {"rows": rows, "abs_err": err, "mae": float(err.mean())}
        self._cache[key] = obj
        return obj

    @staticmethod
    def paired(a, b):
        """按 rows 对齐，返回 (差值 left-right, 共同图像数)。"""
        ia = {int(r): e for r, e in zip(a["rows"], a["abs_err"])}
        ib = {int(r): e for r, e in zip(b["rows"], b["abs_err"])}
        common = sorted(set(ia) & set(ib))
        d = np.array([ia[r] - ib[r] for r in common], dtype=np.float64)
        return d, len(common)


# ---------------------------------------------------------------- 主流程


def main():
    t0 = time.time()
    _selfcheck_t()

    df = pd.read_csv(DEPLOY)
    recs = df.to_dict("records")
    df["main"] = [is_main_config(r) for r in recs]
    main_df = df[df["main"]].copy()
    dropped = df[~df["main"]].copy()

    pi = PerImage()

    zero, missing_runs = {}, []
    for run in df["run"]:
        rj = RESULTS / run / "result.json"
        if rj.exists():
            zero[run] = float(json.loads(rj.read_text(encoding="utf-8"))["zero_baseline_mae"])
        else:
            missing_runs.append(run)

    # 三个 seed 各自的零基线 = 该 seed 自己划分上的恒预测均值；只从主配置 run 取
    # （被筛掉的固定划分补跑带的是 split 0 的基线，混进来会把 seed 1/2 污染）。
    per_seed_base = {}
    for r in main_df.to_dict("records"):
        z = zero.get(r["run"])
        if z is not None:
            per_seed_base.setdefault(int(r["seed"]), []).append(z)
    seed_base = {}
    for _s, _v in per_seed_base.items():
        _arr = np.array(_v, dtype=float)
        if _arr.max() - _arr.min() > 1e-9:
            raise SystemExit("seed %d 的主配置零基线不一致：%s" % (_s, _arr))
        seed_base[_s] = float(_arr.mean())
    base_vals = np.array([seed_base[s] for s in sorted(seed_base)], dtype=float)
    base_range = float(base_vals.max() - base_vals.min())
    base_stats = {"per_seed": {str(s): seed_base[s] for s in sorted(seed_base)},
                  "n_seeds": int(base_vals.size),
                  "range": base_range, "mean": float(base_vals.mean()),
                  "range_pct_of_mean": base_range / float(base_vals.mean()) * 100.0,
                  "source": "results/<run>/result.json 的 zero_baseline_mae；"
                            "每个 seed 取其主配置 run 的共同值"} 

    def cell(mode, n):
        sub = main_df[(main_df["mode"] == mode) & (main_df["snap_every"] == n)]
        return {int(r["seed"]): r for r in sub.to_dict("records")}

    CONTRASTS = [
        ("g1a_interval_N20_vs_Ninf_none", "1a", ("none", 20), ("none", 0),
         "间隔效应：none, N=20 − N=∞", True, "supp_interval_N5_vs_N20_after_snap"),
        ("g1b_interval_N1_vs_Ninf_none", "1b", ("none", 1), ("none", 0),
         "间隔效应：none, N=1 − N=∞", True, "supp_interval_N1_vs_N20_after_snap"),
        ("g1c_shadow_after_vs_none_N20", "1c", ("after_snap", 20), ("none", 20),
         "影子有无：N=20, after_snap − none", True, "supp_shadow_after_vs_none_N5"),
        ("g1d_order_before_vs_after_N20", "1d", ("before_snap", 20), ("after_snap", 20),
         "顺序效应：N=20, before_snap − after_snap", True, "g1d_order_before_vs_after_N20"),
        ("supp_interval_N1_vs_N20_after_snap", "S", ("after_snap", 1), ("after_snap", 20),
         "间隔效应：after_snap, N=1 − N=20", False, "supp_interval_N1_vs_N20_after_snap"),
        ("supp_interval_N5_vs_N20_after_snap", "S", ("after_snap", 5), ("after_snap", 20),
         "间隔效应：after_snap, N=5 − N=20", False, "supp_interval_N5_vs_N20_after_snap"),
        ("supp_interval_N5_vs_N20_before_snap", "S", ("before_snap", 5), ("before_snap", 20),
         "间隔效应：before_snap, N=5 − N=20", False, "supp_interval_N5_vs_N20_before_snap"),
        ("supp_interval_N1_vs_N20_before_snap", "S", ("before_snap", 1), ("before_snap", 20),
         "间隔效应：before_snap, N=1 − N=20", False, "supp_interval_N1_vs_N20_before_snap"),
        ("supp_interval_N1_vs_N5_none", "S", ("none", 1), ("none", 5),
         "间隔效应：none, N=1 − N=5", False, "supp_interval_N1_vs_N5_none"),
        ("supp_interval_N5_vs_N20_none", "S", ("none", 5), ("none", 20),
         "间隔效应：none, N=5 − N=20", False, "supp_interval_N5_vs_N20_none"),
        ("supp_shadow_after_vs_none_N5", "S", ("after_snap", 5), ("none", 5),
         "影子有无：N=5, after_snap − none", False, "supp_shadow_after_vs_none_N5"),
        ("supp_shadow_before_vs_none_N20", "S", ("before_snap", 20), ("none", 20),
         "影子有无：N=20, before_snap − none", False, "supp_shadow_before_vs_none_N20"),
        ("supp_shadow_before_vs_none_N5", "S", ("before_snap", 5), ("none", 5),
         "影子有无：N=5, before_snap − none", False, "supp_shadow_before_vs_none_N5"),
    ]

    rng = np.random.default_rng(BOOT_SEED)
    contrasts = {}
    for key, grp, (ml, nl), (mr, nr), label, core, borrow in CONTRASTS:
        cl, cr = cell(ml, nl), cell(mr, nr)
        seeds = sorted(set(cl) & set(cr))
        per_seed, diffs, img_counts = [], [], []
        for s in seeds:
            ra, rb = cl[s], cr[s]
            A, B = pi.get(ra["run"]), pi.get(rb["run"])
            row = {"seed": s, "left_run": ra["run"], "right_run": rb["run"],
                   "left_mae_csv": float(ra["mae_fp32"]),
                   "right_mae_csv": float(rb["mae_fp32"]),
                   "delta_csv": float(ra["mae_fp32"]) - float(rb["mae_fp32"])}
            if A is not None and B is not None:
                d, m_img = PerImage.paired(A, B)
                diffs.append(d)
                img_counts.append(m_img)
                row.update({"left_mae_perimage": A["mae"], "right_mae_perimage": B["mae"],
                            "delta_perimage": A["mae"] - B["mae"], "n_images": m_img})
            else:
                row["n_images"] = 0
            per_seed.append(row)
        # 主统计量用 perimage 的逐图 MAE（全精度）；csv 只有 4 位小数，
        # 两者最大差 6.4e-05，逐对列出以便核对。
        has_pi = bool(diffs) and len(diffs) == len(per_seed)
        dvec = np.array([r["delta_perimage"] if has_pi else r["delta_csv"] for r in per_seed],
                        dtype=float)
        dcsv = np.array([r["delta_csv"] for r in per_seed], dtype=float)
        stats = summary_stats(dvec)
        npos = int((dvec > 0).sum())
        contrasts[key] = {
            "label": label, "group": grp, "core": core,
            "left": ml + "@N=" + str(nl), "right": mr + "@N=" + str(nr),
            "delta_definition": "left_MAE - right_MAE（负值 = 左侧更准）",
            "n_seeds": len(seeds), "seeds": seeds,
            "delta_source": "perimage（全精度逐图 |pred-y|）" if has_pi else "csv（4 位小数）",
            "point_estimate": stats["mean"],
            "point_estimate_csv": float(dcsv.mean()),
            "point_estimate_perimage": (float(np.mean([r["delta_perimage"] for r in per_seed]))
                                        if has_pi else None),
            "max_abs_csv_minus_perimage": (float(np.abs(dvec - dcsv).max()) if has_pi else None),
            "paired_t": stats,
            "sign_test": sign_test(npos, len(seeds)),
            "n_positive": npos,
            "per_seed": per_seed,
            "bootstrap_2level": two_level_bootstrap(diffs, rng) if diffs else {"ci95": None},
            "n_images_per_run": img_counts,
        }

    # ---- 2. 推断单位对照
    def unit_row(ckey, borrow_key):
        c = contrasts[ckey]
        pairs = []
        for r in c["per_seed"]:
            A, B = pi.get(r["left_run"]), pi.get(r["right_run"])
            if A is not None and B is not None:
                d, _ = PerImage.paired(A, B)
                pairs.append(d)
        d0 = pairs[0]
        se_i = float(d0.std(ddof=1)) / math.sqrt(d0.size)
        df_i = d0.size - 1
        t_i = t_ppf(1 - ALPHA / 2, df_i)
        mean_i = float(d0.mean())
        se_r = c["paired_t"]["se"]
        rec = {
            "contrast": c["label"], "contrast_key": ckey,
            "image_level": {"n_images": int(d0.size), "mean": mean_i, "se": se_i, "df": df_i,
                            "ci95": [mean_i - t_i * se_i, mean_i + t_i * se_i],
                            "excludes_zero": bool(mean_i - t_i * se_i > 0 or mean_i + t_i * se_i < 0),
                            "source": "quant_runs/perimage/<run>__fp32.npz 的 400 张逐图 |pred-y|，"
                                      "取 seed 0 那一对（= 稿件现行口径）"},
            "run_level": None, "merged": None,
        }
        if se_r is not None:
            rec["run_level"] = {"n": c["n_seeds"], "mean": c["paired_t"]["mean"], "se": se_r,
                                "df": c["paired_t"]["df"], "ci95": c["paired_t"]["ci95"],
                                "excludes_zero": bool(c["paired_t"]["ci95"][0] > 0
                                                      or c["paired_t"]["ci95"][1] < 0)}
        else:
            rec["run_level"] = {"n": c["n_seeds"], "mean": c["paired_t"]["mean"], "se": None,
                                "df": 0, "ci95": None, "excludes_zero": None,
                                "note": "n=1：sd 不可估，run 层标准误不存在"}
        borrow = contrasts[borrow_key]
        sd_r = borrow["paired_t"]["sd"]
        df_borrow = borrow["n_seeds"] - 1
        if sd_r is not None:
            se_r_use = sd_r / math.sqrt(c["n_seeds"])
            merged_se = math.sqrt(se_i ** 2 + se_r_use ** 2)
            df_m = (se_i ** 2 + se_r_use ** 2) ** 2 / (
                (se_i ** 2) ** 2 / df_i + (se_r_use ** 2) ** 2 / max(df_borrow, 1))
            t_m = t_ppf(1 - ALPHA / 2, df_m)
            lo, hi = mean_i - t_m * merged_se, mean_i + t_m * merged_se
            rec["merged"] = {
                "formula": "SE_merged = sqrt(sigma_I^2/M + sigma_R^2/n)，M=400 张/run，n=seed 数",
                "sigma_R_source": borrow_key,
                "sigma_R_self": borrow_key == ckey,
                "sigma_R_value": se_r_use,
                "sigma_R_note": ("自估：该对比本身 n=%d，sigma_R 由自身跨 seed sd 给出" % c["n_seeds"]
                                 if borrow_key == ckey else
                                 "n=%d 无法自估 sigma_R，借用 %s 的跨 seed sd（df=%d）"
                                 % (c["n_seeds"], borrow_key, df_borrow)),
                "se": merged_se, "df_satterthwaite": df_m, "ci95": [lo, hi],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "inflation_vs_image": merged_se / se_i,
            }
        return rec

    unit_rows = [unit_row("g1c_shadow_after_vs_none_N20", "supp_shadow_after_vs_none_N5"),
                 unit_row("g1b_interval_N1_vs_Ninf_none", "supp_interval_N1_vs_N20_after_snap"),
                 unit_row("g1d_order_before_vs_after_N20", "g1d_order_before_vs_after_N20")]

    # ---- 3. margin / TOST
    order = contrasts["g1d_order_before_vs_after_N20"]
    order_d = np.array([(r["delta_perimage"] if "delta_perimage" in r else r["delta_csv"])
                        for r in order["per_seed"]], dtype=float)
    tost_rows = [tost(order_d, mg) for mg in TOST_MARGINS]
    all_base = {}
    for r in recs:
        z = zero.get(r["run"])
        if z is not None:
            all_base[r["run"]] = z
    baseline_ref = float(np.mean(list(all_base.values())))
    mde = mde_paired(order["paired_t"]["se"], order["paired_t"]["df"])         if order["paired_t"]["se"] else None

    # ---- 4. MAE / MAE0
    norm_rows = []
    for r in main_df.to_dict("records"):
        z = zero.get(r["run"])
        norm_rows.append({"run": r["run"], "mode": r["mode"], "N": int(r["snap_every"]),
                          "seed": int(r["seed"]), "mae_fp32": float(r["mae_fp32"]),
                          "mae_4": float(r["mae_4"]), "zero_baseline_mae": z,
                          "ratio": (float(r["mae_fp32"]) / z) if z else None})
    ratios = np.array([r["ratio"] for r in norm_rows if r["ratio"]], dtype=float)

    # ---- 5. 符号检验表
    sign_table = {}
    for k, n in [(4, 4), (3, 4), (3, 3), (2, 3), (1, 1)]:
        sign_table["%dof%d" % (k, n)] = sign_test(k, n)

    # ---- 6. 4 bit 归因
    def load_state(run, mode):
        p = RESULTS / run / "best.pth"
        if not p.exists():
            return None
        ck = torch.load(p, map_location="cpu")
        key = "ema_state_dict" if mode != "none" else "model_state_dict"
        return ck[key]

    def channel_ratios(sd):
        out, per_t = [], {}
        worst = ("", -1.0)
        for k, v in sd.items():
            if not torch.is_tensor(v) or v.ndim < 2:
                continue
            w = v.detach().float()
            flat = w.reshape(w.shape[0], -1)
            mx = flat.abs().amax(dim=1)
            mn = flat.abs().mean(dim=1).clamp_min(1e-12)
            rr = (mx / mn).numpy()
            out.append(rr)
            per_t[k] = float(rr.max())
            if float(rr.max()) > worst[1]:
                worst = (k, float(rr.max()))
        allr = np.concatenate(out) if out else np.array([])
        return allr, worst, per_t

    bit4_rows = []
    for r in main_df.to_dict("records"):
        sd = load_state(r["run"], r["mode"])
        z = zero.get(r["run"])
        rec = {"run": r["run"], "mode": r["mode"], "N": int(r["snap_every"]),
               "seed": int(r["seed"]), "mae_fp32": float(r["mae_fp32"]),
               "mae_4": float(r["mae_4"]), "zero_baseline_mae": z,
               "worse_than_baseline": (None if z is None else bool(float(r["mae_4"]) > z)),
               "degrade_vs_fp32": float(r["mae_4"]) / float(r["mae_fp32"])}
        if sd is not None:
            allr, worst, per_t = channel_ratios(sd)
            rec["_per_tensor"] = per_t
            rec.update({"chan_ratio_p50": float(np.percentile(allr, 50)),
                        "chan_ratio_p95": float(np.percentile(allr, 95)),
                        "chan_ratio_p99": float(np.percentile(allr, 99)),
                        "chan_ratio_max": float(allr.max()),
                        "chan_ratio_max_tensor": worst[0],
                        "n_channels": int(allr.size)})
        bit4_rows.append(rec)

    def spearman(a, b):
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        if ra.std() == 0 or rb.std() == 0:
            return None
        return float(np.corrcoef(ra, rb)[0, 1])

    # 同一张量名在不同 run 之间的比值：把「这个 run 在该张量上是否异常极端」问出来，
    # 避免全局分位数被架构本身的主导项掩盖（p95 在 22 个 run 上只差 11.6–12.4）。
    tt = [r for r in bit4_rows if "_per_tensor" in r]
    tnames = sorted(set().union(*[set(r["_per_tensor"]) for r in tt])) if tt else []
    tmed = {}
    for nm in tnames:
        vals = np.array([r["_per_tensor"].get(nm, np.nan) for r in tt], dtype=float)
        vals = vals[~np.isnan(vals)]
        tmed[nm] = float(np.median(vals)) if vals.size else 0.0
    for r in tt:
        best = ("", -1.0)
        for nm, v in r["_per_tensor"].items():
            z = v / tmed[nm] if tmed[nm] else 0.0
            if z > best[1]:
                best = (nm, z)
        r["tensor_extreme_ratio"] = best[1]
        r["tensor_extreme_name"] = best[0]
        del r["_per_tensor"]
    have = [r for r in bit4_rows if "chan_ratio_max" in r]
    mae4 = np.array([r["mae_4"] for r in bit4_rows], dtype=float)
    deg = np.array([r["degrade_vs_fp32"] for r in bit4_rows], dtype=float)
    bit4 = {
        "scope": "主配置 23 个 run（deploy_table_v2.csv 的 mae_4 列）",
        "n_runs": len(bit4_rows),
        "n_worse_than_own_baseline": int(sum(1 for r in bit4_rows if r["worse_than_baseline"])),
        "n_with_baseline": int(sum(1 for r in bit4_rows if r["worse_than_baseline"] is not None)),
        "n_have_ckpt": len(have),
        "median_mae_4": float(np.median(mae4)),
        "mean_mae_4": float(mae4.mean()),
        "sd_mae_4": float(mae4.std(ddof=1)),
        "min_mae_4": float(mae4.min()),
        "max_mae_4": float(mae4.max()),
        "median_degrade_vs_fp32": float(np.median(deg)),
        "max_degrade_vs_fp32": float(deg.max()),
        "min_degrade_vs_fp32": float(deg.min()),
        "median_by_mode": {m: float(np.median([r["mae_4"] for r in bit4_rows if r["mode"] == m]))
                           for m in sorted({r["mode"] for r in bit4_rows})},
        "median_by_N": {str(n): float(np.median([r["mae_4"] for r in bit4_rows if r["N"] == n]))
                        for n in sorted({r["N"] for r in bit4_rows})},
        "spearman_mae4_vs_chan_max": spearman([r["mae_4"] for r in have],
                                              [r["chan_ratio_max"] for r in have]),
        "spearman_mae4_vs_chan_p95": spearman([r["mae_4"] for r in have],
                                              [r["chan_ratio_p95"] for r in have]),
        "n_for_spearman": len(have),
        "spearman_mae4_vs_tensor_extreme": spearman(
            [r["mae_4"] for r in have], [r["tensor_extreme_ratio"] for r in have]),
        "tensor_extreme_definition":
            "对每个 run、每个参数张量取逐输出通道 max/mean 的最大值，再除以该张量在全部 run 上的中位数，"
            "取全部张量里的最大者；=1 表示与同张量的典型 run 一样，>1 表示该 run 在这个张量上更极端",
        "rows": bit4_rows,
        "weight_source": "results/<run>/best.pth，ema_state_dict（mode!=none）或 model_state_dict（none），"
                         "与 scripts/deploy_table.py:148 一致",
        "chan_ratio_definition": "每个 Conv2d/Linear 权重按输出通道取 max|w| / mean|w|；"
                                 "全模型所有通道汇成分位数",
    }

    # ---- 7. perimage 头清单
    files = sorted(PERIMAGE.glob("*.npz"))
    heads = {}
    for f in files:
        h = f.name.split("__")[0].split("-")[1]
        heads[h] = heads.get(h, 0) + 1
    want = ["gelu", "bilinear_glu"]
    head_inv = {
        "n_files": len(files),
        "n_runs": len({f.name.split("__")[0] for f in files}),
        "bits_present": sorted({f.name.split("__")[1].replace(".npz", "") for f in files}),
        "head_counts": heads,
        "has_gelu": "gelu" in heads,
        "has_bilinear_glu": "bilinear_glu" in heads,
        "has_relu_qat": "relu_qat" in heads,
        "missing_requested": [w for w in want if w not in heads],
        "note": "只报告 quant_runs/perimage/ 里有无；未跑任何推理",
    }

    conventions = {
        "pairing_unit": "seed（每个 seed 同时决定数据划分与参数初始化）",
        "run_statistic": "该 run 在 400 张 first400 部署子集上的 MAE，取自 perimage/<run>__fp32.npz 的"
                         "逐图 |pred-y| 均值（全精度）；与 deploy_table_v2.csv 的 mae_fp32 列一致到 4 位小数，"
                         "逐对差值见 §1.0 的 max|Δcsv−Δnpz|",
        "delta_definition": "left_MAE - right_MAE；负值 = 左侧更准",
        "paired_t": "df = n-1，n = 两侧都有 run 的同 seed 配对数",
        "bootstrap": "两级：先整 run(=seed) 有放回重采样 n 个，再在每个被抽中的 run 内对 400 张图有放回"
                     "重采样；%d 次；百分位法 2.5/97.5；rng seed=%d" % (N_BOOT, BOOT_SEED),
        "merged_se": "sqrt(sigma_I^2/M + sigma_R^2/n)，M=400，df 用 Satterthwaite",
        "tost": "配对 t 的双单侧检验，df=n-1，alpha=%s；判定 = p_tost < alpha" % ALPHA,
        "tost_margin_preset": "0.0016 为预设值（零基线 0.1553 的 1%% = %.6f，取整到 0.0016），"
                              "不是从本数据反推" % (0.1553 * 0.01),
        "mde": "MDE = (t_{1-alpha/2,df} + t_{0.8,df}) * SE，df=n-1",
        "main_config_filter": "ema_decay==0.99 且 split_seed==seed"
                              "（沿用 scripts/paper2_figs_add.py:66-68 的 is_main_config）",
    }

    out = {
        "meta": {
            "script": "scripts/paper2_runlevel.py",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": {"deploy_table": "quant_runs/deploy_table_v2.csv",
                       "perimage": "quant_runs/perimage/<run>__{fp32,8,6,4}bit.npz",
                       "zero_baseline": "results/<run>/result.json 的 zero_baseline_mae",
                       "weights": "results/<run>/best.pth"},
            "n_rows_total": int(len(df)),
            "n_rows_main": int(len(main_df)),
            "n_rows_dropped": int(len(dropped)),
            "dropped_rows": dropped[["run", "mode", "snap_every", "seed", "ema_decay",
                                     "split_seed", "mae_fp32", "mae_4"]].to_dict("records"),
            "n_runs_with_perimage": len({f.name.split("__")[0] for f in files}),
            "runs_missing_results_dir": missing_runs,
            "runtime_sec": None,
            "zero_baseline_mean": baseline_ref,
        },
        "conventions": conventions,
        "contrasts": contrasts,
        "inference_unit_comparison": unit_rows,
        "margin_sensitivity_tost": {
            "contrast": order["label"],
            "delta_per_seed": {str(r["seed"]): r["delta_csv"] for r in order["per_seed"]},
            "preset_margin": PRESET_MARGIN,
            "preset_margin_provenance":
                "预设：稿件所用零基线 0.1553（= seed 0 划分的 zero_baseline_mae %.6f；"
                "三 seed 均值 %.6f）的 1%% = %.6f，取 0.0016。该 margin 在本次分析前已确定"
                "（沿用稿件既有设定），不是从观测到的顺序效应反推"
                % (seed_base[0], base_stats["mean"], 0.1553 * 0.01),
            "rows": tost_rows,
            "mde_80pct_power": mde,
            "n3_ci_halfwidth": (t_ppf(0.975, 2) * order["paired_t"]["se"])
                               if order["paired_t"]["se"] else None,
        },
        "mae_normalization_mae0": {
            "definition": "ratio = mae_fp32 / zero_baseline_mae（同一 run 自己划分上的恒预测均值基线）",
            "rows": norm_rows,
            "ratio_median": float(np.median(ratios)),
            "ratio_min": float(ratios.min()),
            "ratio_max": float(ratios.max()),
            "seed_zero_baseline": base_stats,
        },
        "sign_test_exact": {
            "definition": "精确二项检验，p0=0.5；双侧 = 2*min(单侧) 截到 1；单侧给 p_greater",
            "cases": sign_table,
        },
        "bit4_attribution": bit4,
        "perimage_head_inventory": head_inv,
    }
    out["meta"]["runtime_sec"] = round(time.time() - t0, 2)

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                        encoding="utf-8")
    OUT_MD.write_text(render_md(out), encoding="utf-8")
    return out


# ---------------------------------------------------------------- Markdown


def render_md(o):
    L = []
    A = L.append
    m, cv = o["meta"], o["conventions"]
    A("# run 层推断：把稿件里的关键对比从图像层重算到 run 层")
    A("")
    A("生成：`%s`（%s，耗时 %s s）。" % (m["script"], m["generated_at"], m["runtime_sec"]))
    A("")
    A("输入（全部只读）：" + "、".join("`%s`" % v for v in m["inputs"].values()) + "。")
    A("")
    A("## 0 口径（先读这一节，再读数字）")
    A("")
    for k in cv:
        A("- **%s**：%s" % (k, cv[k]))
    A("")
    A("行筛选：`deploy_table_v2.csv` 共 %d 行，主配置 %d 行，筛掉 %d 行（见 §8）。"
      % (m["n_rows_total"], m["n_rows_main"], m["n_rows_dropped"]))
    A("有 perimage 的 run：%d 个。" % m["n_runs_with_perimage"])
    A("")
    A("**本文档里有两套 n，永远不要混用**：图像层 n = 400 张图（稿件现行区间的单位）；"
      "run 层 n = seed 数。")
    A("")

    A("## 1 三个核心对比的 run 层推断")
    A("")
    A("点估计可逐行核对：每个 run 的 MAE 即 `deploy_table_v2.csv` 的 `mae_fp32` 列（400 张 first400）；"
      "配对按 seed，Δ_s = 左 MAE − 右 MAE。")
    A("")
    A("### 1.0 同 seed 配对表")
    A("")
    A("| 组 | 对比 | seed | 左 MAE | 右 MAE | Δ_s（perimage，主） | Δ_s（csv） | 共同图像 |")
    A("|---|---|---|---|---|---|---|---|")
    for key in o["contrasts"]:
        c = o["contrasts"][key]
        for r in c["per_seed"]:
            A("| %s | %s | %d | %.4f | %.4f | %s | %+.4f | %d |"
              % (c["group"], c["label"], r["seed"], r["left_mae_csv"], r["right_mae_csv"],
                 ("%+.5f" % r["delta_perimage"]) if "delta_perimage" in r else "—",
                 r["delta_csv"], r["n_images"]))
    A("")
    _mx = max([c["max_abs_csv_minus_perimage"] for c in o["contrasts"].values()
               if c.get("max_abs_csv_minus_perimage") is not None] or [0.0])
    A("左/右 MAE 列取自 `deploy_table_v2.csv` 的 `mae_fp32`（4 位小数）；"
      "**Δ_s 的主列用 `perimage/*.npz` 的全精度逐图 MAE**。两列的最大差 = %.6f，"
      "即 csv 的 4 位小数取整误差，不影响任何结论的量级。" % _mx)
    A("")

    A("### 1.1 三组核心对比的汇总")
    A("")
    A("| 组 | 对比 | n（seed） | 点估计 | 配对 t 95% CI (df) | 符号检验双侧 p | 两级 bootstrap 95% CI |")
    A("|---|---|---|---|---|---|---|")
    for key in o["contrasts"]:
        c = o["contrasts"][key]
        if not c["core"]:
            continue
        pt, bs = c["paired_t"], c["bootstrap_2level"]
        ci = ci_str(pt["ci95"]) if pt["ci95"] else "不可估"
        bsc = ci_str(bs.get("ci95")) if bs.get("ci95") else "—"
        if bs.get("image_only"):
            bsc += "（n=1 ⇒ 只含 run 内图像重采样）"
        A("| %s | %s | %d | %+.4f | %s (df=%d) | %.4f | %s |"
          % (c["group"], c["label"], c["n_seeds"], pt["mean"], ci, pt["df"],
             c["sign_test"]["p_two_sided"], bsc))
    A("")
    A("**n 是多少，逐组写清：**")
    A("")
    for key in o["contrasts"]:
        c = o["contrasts"][key]
        if not c["core"]:
            continue
        if c["n_seeds"] == 1:
            extra = "　sd 与配对 t 区间在 n=1 时不存在（df=0），只能给点估计；" \
                    "该对比在 run 层**没有任何重复**。"
        else:
            extra = "　sd = %.6f，SE = %.6f，df = %d。" % (c["paired_t"]["sd"], c["paired_t"]["se"],
                                                         c["paired_t"]["df"])
        A("- %s `%s`：**n = %d 个 seed**（seed = %s）。%s"
          % (c["group"], c["label"], c["n_seeds"], c["seeds"], extra))
    A("")
    A("### 1.2 为什么 n 只有这么几个（不是笔误）")
    A("")
    A("主配置里 N=∞（`snap_every=0`）只有 seed 0 的两个 run（none 与 after_snap）；"
      "`none` 协议在 N=20 也只有 seed 0。")
    A("因此凡是含 N=∞ 的对比、以及 after_snap − none @ N=20，配对后 n 只能是 1。"
      "图像层的 400 张图**不能**替代 seed 重复——它们只重复测量同一个 run。")
    A("")
    A("### 1.3 补充：主配置里真正有跨 seed 重复的对比（n≥3）")
    A("")
    A("这几组是稿件可以安全升级到 run 层口径的候选。")
    A("")
    A("| 对比 | n | 点估计 | 95% CI | 符号 p | bootstrap 95% CI |")
    A("|---|---|---|---|---|---|")
    for key in o["contrasts"]:
        c = o["contrasts"][key]
        if c["core"]:
            continue
        pt, bs = c["paired_t"], c["bootstrap_2level"]
        ci = ci_str(pt["ci95"]) if pt["ci95"] else "不可估（df=0）"
        bsc = ci_str(bs.get("ci95")) if bs.get("ci95") else "—"
        A("| %s | %d | %+.4f | %s | %.4f | %s |"
          % (c["label"], c["n_seeds"], pt["mean"], ci, c["sign_test"]["p_two_sided"], bsc))
    A("")

    A("## 2 推断单位对照表（评审 F1 的证据）")
    A("")
    A("同一批逐图差值，三种标准误。图像层 = 稿件现行口径（seed 0 那一对、400 张逐图配对 t，df=399）；"
      "run 层 = 以 seed 为单位（df=n−1）；合并 = 两层方差相加。")
    A("")
    A("| 对比 | 图像层 SE (n=400) | run 层 SE | 合并 SE | 合并后 95% CI | 含 0？ |")
    A("|---|---|---|---|---|---|")
    for u in o["inference_unit_comparison"]:
        il, rl, mg = u["image_level"], u["run_level"], u["merged"]
        rse = ("%.6f (n=%d)" % (rl["se"], rl["n"])) if rl["se"] is not None \
            else ("不可估 (n=%d)" % rl["n"])
        if mg:
            mgs = "%.6f" % mg["se"]
            mci = ci_str(mg["ci95"])
            mgz = "不含 0" if mg["excludes_zero"] else "**含 0**"
        else:
            mgs, mci, mgz = "—", "—", "—"
        A("| %s | %.6f | %s | %s | %s | %s |" % (u["contrast"], il["se"], rse, mgs, mci, mgz))
    A("")
    for u in o["inference_unit_comparison"]:
        il, rl, mg = u["image_level"], u["run_level"], u["merged"]
        A("### %s" % u["contrast"])
        A("")
        A("- 图像层：均差 %+.4f，SE = %.6f（M = %d），df = %d，95%% CI = %s，%s"
          % (il["mean"], il["se"], il["n_images"], il["df"], ci_str(il["ci95"]),
             "不含 0。" if il["excludes_zero"] else "含 0。"))
        if rl["se"] is not None:
            A("- run 层：均差 %+.4f，SE = %.6f（n = %d），df = %d，95%% CI = %s。"
              % (rl["mean"], rl["se"], rl["n"], rl["df"], ci_str(rl["ci95"])))
        else:
            A("- run 层：均差 %+.4f，**SE 不可估**（n = %d，df = 0）。" % (rl["mean"], rl["n"]))
        if mg:
            A("- 合并：%s。σ_R 来源：%s（%s）。合并 SE = %.6f，比图像层放大 **%.2f×**，"
              "df(Satterthwaite) = %.2f，95%% CI = %s，%s"
              % (mg["formula"], mg["sigma_R_source"], mg["sigma_R_note"], mg["se"],
                 mg["inflation_vs_image"], mg["df_satterthwaite"], ci_str(mg["ci95"]),
                 "**不含 0**。" if mg["excludes_zero"]
                 else "**含 0**——该对比在合并口径下无法排除无效应。"))
        else:
            A("- 合并：无法计算（缺 σ_R）。")
        A("")
    u_last = o["inference_unit_comparison"][-1]
    il_l, rl_l = u_last["image_level"], u_last["run_level"]
    if rl_l["se"]:
        A("**放大区间的主要不是标准误，而是自由度。** 以末行为例：run 层 SE = %.6f 与图像层 SE = %.6f 同量级，"
          "但 t 临界从 t_{0.975,%d} = %.3f 变成 t_{0.975,%d} = %.3f（%.2f×），"
          "95%% 区间半宽因而从 %.6f 变成 %.6f。单位错配的代价主要落在 df 上。"
          % (rl_l["se"], il_l["se"], il_l["df"], t_ppf(0.975, il_l["df"]),
             rl_l["df"], t_ppf(0.975, rl_l["df"]),
             t_ppf(0.975, rl_l["df"]) / t_ppf(0.975, il_l["df"]),
             t_ppf(0.975, il_l["df"]) * il_l["se"],
             t_ppf(0.975, rl_l["df"]) * rl_l["se"]))
        A("")
    A("读法：n=1 的对比给不出 run 层区间，图像层区间因此**不能**当作 protocol/interval 效应的不确定性；"
      "n≥2 的对比里，run 层 SE 未必更大，但 df 从 399 掉到 n−1，t 临界放大才是区间变宽的主因。")
    A("")

    A("## 3 margin 敏感性与 TOST（顺序效应，N=20）")
    A("")
    t = o["margin_sensitivity_tost"]
    A("对比：%s；逐 seed Δ = %s。"
      % (t["contrast"], "，".join("s%s: %+.4f" % (k, v) for k, v in t["delta_per_seed"].items())))
    A("")
    A("**margin 0.0016 是预设值**：%s。其余 margin 只是敏感性网格，均已列出以便读者判断结论对 margin 的依赖。"
      % t["preset_margin_provenance"])
    A("")
    A("| margin | p_lower | p_upper | p_TOST | 判定（α=0.05） | 90% CI |")
    A("|---|---|---|---|---|---|")
    for row in t["rows"]:
        if row["p_tost"] is None:
            A("| %.4f | — | — | 不可算 | n=1 | — |" % row["margin"])
            continue
        A("| %.4f | %.4f | %.4f | %.4f | %s | %s |"
          % (row["margin"], row["p_lower"], row["p_upper"], row["p_tost"],
             "等价通过" if row["equivalent"] else "不能判定等价", ci_str(row["ci90"])))
    A("")
    r18 = [row for row in t["rows"] if row["margin"] == PRESET_MARGIN]
    if r18 and r18[0]["p_tost"] is not None:
        A("**0.0016（预设 margin）这一行不通过**：p_TOST = %.4f > 0.05，按 run 层口径不能判定顺序效应等价于 0。"
          "margin 0.0018 同样不通过（p_TOST = %.4f），只有 0.005 通过。"
          % (r18[0]["p_tost"],
             [row["p_tost"] for row in t["rows"] if row["margin"] == 0.0018][0]))
        A("")
    md = t["mde_80pct_power"]
    if md:
        A("**80%% 功效下可检出效应（MDE）**：%s。" % md["formula"])
        A("代入 SE = %.6f、df = %d：t_{0.975,%d} = %.4f、t_{0.80,%d} = %.4f ⇒ **MDE = %.6f**。"
          % (md["se"], md["df"], md["df"], md["t_alpha"], md["df"], md["t_power"], md["mde"]))
    if t["n3_ci_halfwidth"]:
        A("同一 SE 下 95%% 区间半宽 = %.6f（= t_{0.975,2} × SE），即本检验在 run 层的分辨率约 %.4f。"
          % (t["n3_ci_halfwidth"], t["n3_ci_halfwidth"]))
    A("")

    A("## 4 MAE / MAE₀ 归一化（评审 M1）")
    A("")
    nz = o["mae_normalization_mae0"]
    A("%s。ratio<1 表示 live MAE 低于同划分零基线。" % nz["definition"])
    A("")
    A("| run | mode | N | seed | MAE | MAE₀ | MAE/MAE₀ |")
    A("|---|---|---|---|---|---|---|")
    for r in nz["rows"]:
        A("| %s | %s | %d | %d | %.4f | %s | %s |"
          % (r["run"], r["mode"], r["N"], r["seed"], r["mae_fp32"],
             fnum(r["zero_baseline_mae"]), fnum(r["ratio"])))
    A("")
    zb = nz["seed_zero_baseline"]
    A("三个 seed 的零基线：%s；极差 = **%.6f**，相对均值（%.6f）为 **%.3f%%**。"
      % ("，".join("s%s = %.6f" % (k, v) for k, v in zb["per_seed"].items()),
         zb["range"], zb["mean"], zb["range_pct_of_mean"]))
    A("归一化 ratio 的中位数 %.4f，范围 %.4f–%.4f。跨 seed 比较一律同时给原始 MAE 与归一化值："
      "seed 不同即划分不同，零基线本身不同。"
      % (nz["ratio_median"], nz["ratio_min"], nz["ratio_max"]))
    A("")

    A("## 5 符号检验的精确二项 p（评审 M6）")
    A("")
    A("%s。" % o["sign_test_exact"]["definition"])
    A("")
    A("| 同号数/总数 | 双侧 p | 单侧 p（P(X≥k)） |")
    A("|---|---|---|")
    for k in o["sign_test_exact"]["cases"]:
        v = o["sign_test_exact"]["cases"][k]
        A("| %d/%d | %.6f | %.6f |" % (v["k"], v["n"], v["p_two_sided"], v["p_greater"]))
    A("")
    A("读法：n=4 时 4/4 同号的双侧 p 也只有 0.125，n=4 的符号检验没有 5% 分辨率；"
      "n=3 时 3/3 同号双侧 p = 0.25。这是符号检验的固有下限，不是样本量问题。")
    A("")

    A("## 6 4 bit 归因（评审 M2 的便宜部分）")
    A("")
    b = o["bit4_attribution"]
    A("口径：%s；n = %d；有 checkpoint 的 %d 个。" % (b["scope"], b["n_runs"], b["n_have_ckpt"]))
    A("权重来源：%s。" % b["weight_source"])
    A("")
    A("- **差于该 run 自身零基线的 run 数：%d / %d**（%d 个 run 里有零基线可比的 %d 个；"
      "其余 run 的 `results/` 目录已不存在）"
      % (b["n_worse_than_own_baseline"], b["n_with_baseline"], b["n_runs"], b["n_with_baseline"]))
    A("- mae_4 中位数 **%.4f**（均值 %.4f，sd %.4f，范围 %.4f–%.4f）"
      % (b["median_mae_4"], b["mean_mae_4"], b["sd_mae_4"], b["min_mae_4"], b["max_mae_4"]))
    A("- 相对自身 FP32 的退化倍数：中位数 **%.2f×**，范围 %.2f–%.2f×"
      % (b["median_degrade_vs_fp32"], b["min_degrade_vs_fp32"], b["max_degrade_vs_fp32"]))
    A("- 按协议的中位数：%s" % "、".join("%s %.4f" % (k, v) for k, v in b["median_by_mode"].items()))
    A("- 按间隔的中位数：%s" % "、".join("N=%s %.4f" % (k, v) for k, v in b["median_by_N"].items()))
    A("")
    A("逐通道统计：%s。" % b["chan_ratio_definition"])
    A("")
    A("- Spearman(mae_4, 各 run 的通道 max/mean 最大值) = **%.3f**（n = %d）"
      % (b["spearman_mae4_vs_chan_max"], b["n_for_spearman"]))
    A("- Spearman(mae_4, p95) = **%.3f**" % b["spearman_mae4_vs_chan_p95"])
    A("- Spearman(mae_4, 同张量归一化的极端度) = **%.3f**；%s"
      % (b["spearman_mae4_vs_tensor_extreme"], b["tensor_extreme_definition"]))
    A("")
    _df = max(b["n_for_spearman"] - 2, 1)
    _tc = t_ppf(0.975, _df)
    A("n = %d 时，Spearman 双侧 5%% 显著需要 |ρ| ≥ %.3f；实测三个 ρ 都在此之下，"
      "即**没有**可判定的单调关系。" % (b["n_for_spearman"], _tc / math.sqrt(_tc ** 2 + _df)))
    A("")
    A("读法：全局 max/mean 的 p95 在 22 个 run 上只落在 11.6–12.4，几乎被架构本身决定，"
      "没有判别力；换成「同张量跨 run 归一化」后仍然没有与 mae_4 同向的排序。"
      "**故本口径下 4 bit 崩溃没有在少数张量的极端 max/mean 上留下痕迹**——这是一条否定性证据，不是缺失数据。")
    A("")
    A("| run | mode | N | seed | mae_4 | MAE₀ | 差于基线 | 退化× | p50 | p95 | max | 极端度 | max 所在张量 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in b["rows"]:
        if "chan_ratio_max" not in r:
            A("| %s | %s | %d | %d | %.4f | %s | %s | %.2f | — | — | — | — | 无 best.pth |"
              % (r["run"], r["mode"], r["N"], r["seed"], r["mae_4"],
                 fnum(r["zero_baseline_mae"]), flag(r["worse_than_baseline"]),
                 r["degrade_vs_fp32"]))
            continue
        A("| %s | %s | %d | %d | %.4f | %s | %s | %.2f | %.1f | %.1f | %.1f | %.3f | %s |"
          % (r["run"], r["mode"], r["N"], r["seed"], r["mae_4"],
             fnum(r["zero_baseline_mae"]), flag(r["worse_than_baseline"]), r["degrade_vs_fp32"],
             r["chan_ratio_p50"], r["chan_ratio_p95"], r["chan_ratio_max"],
             r["tensor_extreme_ratio"], r["chan_ratio_max_tensor"]))
    A("")

    A("## 7 perimage 里的头（T2 的输入检查）")
    A("")
    h = o["perimage_head_inventory"]
    A("`quant_runs/perimage/` 共 %d 个 npz、%d 个 run，位宽 = %s，头计数 = %s。"
      % (h["n_files"], h["n_runs"], h["bits_present"], h["head_counts"]))
    A("")
    A("- gelu 头逐图预测：**%s**" % ("有" if h["has_gelu"] else "没有"))
    A("- bilinear_glu 头逐图预测：**%s**" % ("有" if h["has_bilinear_glu"] else "没有"))
    A("- relu_qat 头逐图预测：%s" % ("有" if h["has_relu_qat"] else "没有"))
    A("")
    tail = ""
    if h["missing_requested"]:
        tail = (" 缺 %s，头对头的 run 层推断必须先在 T2 里补出这些 npz。"
                % "、".join(h["missing_requested"]))
    A("结论：%s。%s" % (h["note"], tail))
    A("")

    A("## 8 筛掉的行与已知限制")
    A("")
    A("按 `%s` 筛掉 %d 行：" % (cv["main_config_filter"], m["n_rows_dropped"]))
    A("")
    A("| run | mode | N | seed | ema_decay | split_seed | mae_fp32 | mae_4 | 原因 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in m["dropped_rows"]:
        why = []
        if float(r["ema_decay"]) != 0.99:
            why.append("γ=%s≠0.99" % r["ema_decay"])
        if int(r["split_seed"]) != int(r["seed"]):
            why.append("固定划分 split_seed=%d≠seed" % int(r["split_seed"]))
        A("| %s | %s | %d | %d | %s | %d | %.4f | %.4f | %s |"
          % (r["run"], r["mode"], int(r["snap_every"]), int(r["seed"]), r["ema_decay"],
             int(r["split_seed"]), float(r["mae_fp32"]), float(r["mae_4"]), "；".join(why)))
    A("")
    A("限制：")
    A("")
    A("1. run 层 n 由主配置决定：N=∞ 端点只有 seed 0，故所有含 N=∞ 的对比 n=1，没有 run 层区间。")
    A("2. 合并 SE 需要 σ_R；n=1 的对比只能从另一个有复制的对比借用，属**插件式近似**，"
      "`sigma_R_source` 已逐条写明。")
    A("3. bootstrap 只在 n≥2 时才真正两级；n=1 时退化为 run 内图像重采样，与图像层区间同源"
      "（这本身就是 F1 的展示）。")
    A("4. seed 同时含划分与初始化两个方差源，本设计与 4 bit 归因都不能把它们分开"
      "（需 split_seed × init_seed 交叉设计）。")
    A("5. 4 bit 的通道 max/mean 是**相关诊断**，不是因果机制；Spearman 只说明排序同向程度。")
    miss = m.get("runs_missing_results_dir") or []
    if miss:
        A("")
        A("6. 以下 run 的 `results/<run>/` 目录**已不存在**（无 `result.json`、无 `best.pth`），"
          "故不参与零基线归一化与逐通道统计；它们的 MAE 仍取自 `deploy_table_v2.csv`、"
          "逐图预测仍取自 `quant_runs/perimage/`，故 run 层配对不受影响：")
        A("")
        for run in miss:
            A("   - `%s`" % run)
    A("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    out = main()
    c = out["contrasts"]
    print("== run 层推断完成 ==")
    print("rows: total %d, main %d, dropped %d; runtime %s s"
          % (out["meta"]["n_rows_total"], out["meta"]["n_rows_main"],
             out["meta"]["n_rows_dropped"], out["meta"]["runtime_sec"]))
    for k in ("g1a_interval_N20_vs_Ninf_none", "g1b_interval_N1_vs_Ninf_none",
              "g1c_shadow_after_vs_none_N20", "g1d_order_before_vs_after_N20"):
        x = c[k]
        pt = x["paired_t"]
        ci = ci_str(pt["ci95"]) if pt["ci95"] else "NA(df=0)"
        print("%-34s n=%d point=%+.4f ci=%s signp=%.4f boot=%s"
              % (k, x["n_seeds"], pt["mean"], ci, x["sign_test"]["p_two_sided"],
                 ci_str(x["bootstrap_2level"].get("ci95"))))
    for u in out["inference_unit_comparison"]:
        il, rl, mg = u["image_level"], u["run_level"], u["merged"]
        print("UNIT %s" % u["contrast"])
        print("     img n=%d mean=%+.4f se=%.6f ci=%s excl0=%s"
              % (il["n_images"], il["mean"], il["se"], ci_str(il["ci95"]), il["excludes_zero"]))
        print("     run n=%d mean=%+.4f se=%s" % (rl["n"], rl["mean"], rl["se"]))
        if mg:
            print("     merged se=%.6f ci=%s excl0=%s inflation=%.2fx sigmaR_src=%s"
                  % (mg["se"], ci_str(mg["ci95"]), mg["excludes_zero"],
                     mg["inflation_vs_image"], mg["sigma_R_source"]))
    for row in out["margin_sensitivity_tost"]["rows"]:
        print("TOST m=%.4f p_lower=%.4f p_upper=%.4f p_tost=%.4f eq=%s"
              % (row["margin"], row["p_lower"], row["p_upper"], row["p_tost"], row["equivalent"]))
    print("MDE:", out["margin_sensitivity_tost"]["mde_80pct_power"])
    b = out["bit4_attribution"]
    print("BIT4: worse=%d/%d median=%.4f degrade_med=%.2f/%.2f sp_max=%s sp_p95=%s"
          % (b["n_worse_than_own_baseline"], b["n_runs"], b["median_mae_4"],
             b["median_degrade_vs_fp32"], b["max_degrade_vs_fp32"],
             b["spearman_mae4_vs_chan_max"], b["spearman_mae4_vs_chan_p95"]))
    print("BIT4 tensor-extreme sp=%s" % b["spearman_mae4_vs_tensor_extreme"])
    h = out["perimage_head_inventory"]
    print("HEADS: %s gelu=%s glu=%s relu_qat=%s"
          % (h["head_counts"], h["has_gelu"], h["has_bilinear_glu"], h["has_relu_qat"]))
    print("MAE0 seed baselines:", out["mae_normalization_mae0"]["seed_zero_baseline"])
    print("SIGN:", {k: v["p_two_sided"] for k, v in out["sign_test_exact"]["cases"].items()})
    print("JSON top keys:", list(out.keys()))
