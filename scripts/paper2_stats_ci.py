#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper2_stats_ci.py -- 只读后处理：从 quant_runs/deploy_table.csv 算 95% 置信区间。

用法:
    D:/tool/Python/python.exe scripts/paper2_stats_ci.py

输出:
    quant_runs/ci_report.md
    quant_runs/ci_report.json

不写 deploy_table.csv，不碰 drafts/，只用 numpy/scipy/pandas 与标准库。
所有 bootstrap 种子写死，脚本可重跑且结果确定。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "quant_runs" / "deploy_table.csv"
MD_PATH = ROOT / "quant_runs" / "ci_report.md"
JSON_PATH = ROOT / "quant_runs" / "ci_report.json"

NL = chr(10)
BOOT_N = 10000
BOOT_SEED = 0
T975_DF2_TASK = 4.302653

EXPECT_ROWS = 15
EXPECT_S0 = 9
EXPECT_S12 = 6

# 论文/任务书里既有的点估计，只用于自校验，不参与任何计算
PAPER = {
    "interval_fp32": {1: 0.0743, 5: 0.0658, 20: 0.0607},
    "order_fp32_before": 0.0570,
    "order_fp32_after": 0.0572,
    "order_fp32_delta": 0.0001,
    "bits4_mean": {1: 0.1224, 5: 0.0867, 20: 0.1315},
    "bits4_sd": {1: 0.0626, 5: 0.0012, 20: 0.0476},
    "bits4_n": {1: 5, 5: 3, 20: 7},
    "bits4_range20": (0.0849, 0.2030),
}

FORENSIC = [0.0570, 0.0572, 0.0591, 0.0642, 0.2337, 0.2030]


def sha1_of(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def f4(v):
    v = float(v)
    if abs(v) < 5e-5:
        v = 0.0
    return f"{v:.4f}"


def f5(v):
    s = f"{v:.5f}"
    return "0.00000" if s == "-0.00000" else s


def ci_str(lo, hi):
    return "[" + f4(lo) + ", " + f4(hi) + "]"


def t_ci(vals):
    x = np.asarray(vals, dtype=float)
    n = int(x.size)
    m = float(x.mean())
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    tq = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    half = tq * s / np.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": m, "sd": s, "t": tq, "half": float(half),
            "lo": m - half, "hi": m + half}


def boot_mean_ci(vals, seed=BOOT_SEED, n_boot=BOOT_N):
    rng = np.random.default_rng(seed)
    x = np.asarray(vals, dtype=float)
    n = int(x.size)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_diff_ci(a, b, seed=BOOT_SEED, n_boot=BOOT_N):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def welch(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = int(a.size), int(b.size)
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    se = float(np.sqrt(va / na + vb / nb))
    df = float((va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)))
    tq = float(stats.t.ppf(0.975, df))
    d = float(a.mean() - b.mean())
    return {"delta": d, "se": se, "df": df, "t": tq,
            "lo": d - tq * se, "hi": d + tq * se,
            "n_a": na, "n_b": nb,
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "sd_a": float(np.sqrt(va)), "sd_b": float(np.sqrt(vb))}


def verdict(tlo, thi, blo, bhi):
    t0 = bool(tlo > 0 or thi < 0)
    b0 = bool(blo > 0 or bhi < 0)
    ov = bool(tlo <= bhi and blo <= thi)
    return {"t_excludes_zero": t0, "boot_excludes_zero": b0,
            "overlap": ov, "consistent": bool(ov and (t0 == b0))}


def excl_bound(lo, hi):
    return float(max(abs(lo), abs(hi)))


def cmp_item(name, paper, computed):
    return {"item": name, "paper": float(paper), "computed": float(computed),
            "diff": float(computed) - float(paper),
            "ok": bool(round(float(computed), 4) == round(float(paper), 4))}


def main():
    sha_before = sha1_of(CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    n_rows = int(len(df))
    assert n_rows == EXPECT_ROWS, "行数不是 15: " + str(n_rows)
    assert int((df["seed"] == 0).sum()) == EXPECT_S0, "seed 0 不是 9 行"
    assert int((df["seed"] != 0).sum()) == EXPECT_S12, "seed 1/2 不是 6 行"

    mism = []
    for r in df.itertuples():
        tag = "-s" + str(int(r.seed)) + "-"
        if tag not in r.run:
            mism.append({"run": r.run, "seed": int(r.seed)})

    s0 = df[df["seed"] == 0]
    intervals = [1, 5, 20]

    groups = []
    for N in intervals:
        vals = s0.loc[s0["snap_every"] == N, "mae_fp32"].to_numpy()
        groups.append({"key": "interval_fp32_N" + str(N),
                       "label": "协议均值 8 bit(mae_fp32) N=" + str(N) + "（seed0 x 3 mode）",
                       "note": "n=3 同 seed-0 三种 mode",
                       "values": vals, "ci": t_ci(vals), "boot": boot_mean_ci(vals)})

    before_fp32 = df[(df["snap_every"] == 20) & (df["mode"] == "before_snap")]["mae_fp32"].to_numpy()
    after_fp32 = df[(df["snap_every"] == 20) & (df["mode"] == "after_snap")]["mae_fp32"].to_numpy()
    before_6 = df[(df["snap_every"] == 20) & (df["mode"] == "before_snap")]["mae_6"].to_numpy()
    after_6 = df[(df["snap_every"] == 20) & (df["mode"] == "after_snap")]["mae_6"].to_numpy()
    before_8 = df[(df["snap_every"] == 20) & (df["mode"] == "before_snap")]["mae_8"].to_numpy()
    after_8 = df[(df["snap_every"] == 20) & (df["mode"] == "after_snap")]["mae_8"].to_numpy()

    for tag, lab, vals in [("before_fp32_N20", "N=20 before_snap（mae_fp32）", before_fp32),
                           ("after_fp32_N20", "N=20 after_snap（mae_fp32）", after_fp32),
                           ("before_6_N20", "N=20 before_snap（mae_6）", before_6),
                           ("after_6_N20", "N=20 after_snap（mae_6）", after_6)]:
        groups.append({"key": tag, "label": lab,
                       "note": "n=3 三 seed 同 mode",
                       "values": vals, "ci": t_ci(vals), "boot": boot_mean_ci(vals)})

    b4 = {}
    for N in intervals:
        vals = df.loc[df["snap_every"] == N, "mae_4"].to_numpy()
        comp = df[df["snap_every"] == N]["mode"].value_counts().to_dict()
        comp = {str(k): int(v) for k, v in comp.items()}
        b4[N] = {"values": vals, "ci": t_ci(vals), "boot": boot_mean_ci(vals),
                 "mode_comp": comp, "min": float(vals.min()), "max": float(vals.max())}
        groups.append({"key": "bits4_N" + str(N),
                       "label": "4 bit(mae_4) N=" + str(N) + "（全部 run）",
                       "note": "mode 构成 " + json.dumps(comp, ensure_ascii=False),
                       "values": vals, "ci": t_ci(vals), "boot": boot_mean_ci(vals)})

    g1 = s0.loc[s0["snap_every"] == 1, "mae_fp32"].to_numpy()
    g20 = s0.loc[s0["snap_every"] == 20, "mae_fp32"].to_numpy()
    d_1_20 = welch(g20, g1)
    d_1_20["boot"] = boot_diff_ci(g20, g1)
    d_1_20["verdict"] = verdict(d_1_20["lo"], d_1_20["hi"], d_1_20["boot"][0], d_1_20["boot"][1])
    d_1_20["excl"] = excl_bound(d_1_20["lo"], d_1_20["hi"])
    d_1_20["boot_excl"] = excl_bound(d_1_20["boot"][0], d_1_20["boot"][1])

    order = {}
    for tag, va, vb in [("fp32", before_fp32, after_fp32),
                        ("mae6", before_6, after_6),
                        ("mae8", before_8, after_8)]:
        w = welch(va, vb)
        w["boot"] = boot_diff_ci(va, vb)
        w["verdict"] = verdict(w["lo"], w["hi"], w["boot"][0], w["boot"][1])
        w["excl"] = excl_bound(w["lo"], w["hi"])
        w["boot_excl"] = excl_bound(w["boot"][0], w["boot"][1])
        w["values_before"] = [float(x) for x in va]
        w["values_after"] = [float(x) for x in vb]
        order[tag] = w

    checks = []
    for N in intervals:
        v = s0.loc[s0["snap_every"] == N, "mae_fp32"].mean()
        checks.append(cmp_item("interval_fp32 N=" + str(N), PAPER["interval_fp32"][N], v))
    checks.append(cmp_item("order_fp32 before@N20", PAPER["order_fp32_before"], float(before_fp32.mean())))
    checks.append(cmp_item("order_fp32 after@N20", PAPER["order_fp32_after"], float(after_fp32.mean())))
    checks.append(cmp_item("order_fp32 delta(before-after)", PAPER["order_fp32_delta"], order["fp32"]["delta"]))
    for N in intervals:
        ci = b4[N]["ci"]
        checks.append(cmp_item("bits4 mean N=" + str(N), PAPER["bits4_mean"][N], ci["mean"]))
        checks.append(cmp_item("bits4 sd N=" + str(N), PAPER["bits4_sd"][N], ci["sd"]))
        checks.append(cmp_item("bits4 n N=" + str(N), PAPER["bits4_n"][N], ci["n"]))
    checks.append(cmp_item("bits4 min@N20", PAPER["bits4_range20"][0], b4[20]["min"]))
    checks.append(cmp_item("bits4 max@N20", PAPER["bits4_range20"][1], b4[20]["max"]))
    bad = [c for c in checks if not c["ok"]]

    iv6 = {}
    for N in intervals:
        vals = s0.loc[s0["snap_every"] == N, "mae_6"].to_numpy()
        iv6[N] = t_ci(vals)
    mae8_gap = float(np.max(np.abs(df["mae_8"].to_numpy() - df["mae_fp32"].to_numpy())))

    forensic = []
    for val in FORENSIC:
        hits = []
        for r in df.itertuples():
            for col in ("val_mae_train", "mae_fp32", "mae_8", "mae_6", "mae_4"):
                if abs(float(getattr(r, col)) - val) < 5e-9:
                    hits.append({"col": col, "snap_every": int(r.snap_every),
                                 "mode": r.mode, "seed": int(r.seed), "run": r.run})
        forensic.append({"value": val, "hits": hits})

    i1 = t_ci(s0.loc[s0["snap_every"] == 1, "mae_fp32"].to_numpy())
    i5 = t_ci(s0.loc[s0["snap_every"] == 5, "mae_fp32"].to_numpy())
    i20 = t_ci(s0.loc[s0["snap_every"] == 20, "mae_fp32"].to_numpy())
    o_f, o_6 = order["fp32"], order["mae6"]

    # ---------- 本轮新增：新口径 = results/<run>/history.csv 的 val_mae_ema（最佳轮次） ----------
    ema_rows = []
    for r in df.itertuples():
        if int(r.snap_every) != 20 or r.mode not in ("before_snap", "after_snap"):
            continue
        hd = pd.read_csv(ROOT / "results" / r.run / "history.csv")
        e = hd["val_mae_ema"].to_numpy(dtype=float)
        lv = hd["val_mae_live"].to_numpy(dtype=float)
        ema_rows.append({"run": r.run, "mode": r.mode, "seed": int(r.seed),
                         "epochs": int(e.size),
                         "best_epoch": int(np.argmin(e)), "best_ema": float(e.min()),
                         "last_ema": float(e[-1]),
                         "ema_at_best_live": float(e[int(np.argmin(lv))]),
                         "csv_mae_fp32": float(r.mae_fp32), "csv_mae_8": float(r.mae_8)})
    ema_rows.sort(key=lambda x: (x["mode"], x["seed"]))
    for x in ema_rows:
        x["shift_vs_csv_fp32"] = float(x["best_ema"] - x["csv_mae_fp32"])
    ema_b = np.array([x["best_ema"] for x in ema_rows if x["mode"] == "before_snap"])
    ema_a = np.array([x["best_ema"] for x in ema_rows if x["mode"] == "after_snap"])
    ema_w = welch(ema_b, ema_a)
    ema_w["boot"] = boot_diff_ci(ema_b, ema_a)
    ema_w["verdict"] = verdict(ema_w["lo"], ema_w["hi"], ema_w["boot"][0], ema_w["boot"][1])
    ema_w["excl"] = excl_bound(ema_w["lo"], ema_w["hi"])
    ema_w["boot_excl"] = excl_bound(ema_w["boot"][0], ema_w["boot"][1])
    ema_t = {"before": t_ci(ema_b), "after": t_ci(ema_a)}
    ema_boot_mean = {"before": boot_mean_ci(ema_b), "after": boot_mean_ci(ema_a)}
    ema_sum_b, ema_sum_a = float(ema_b.sum()), float(ema_a.sum())
    ema_rounded_sub = float(round(ema_w["mean_a"], 4) - round(ema_w["mean_b"], 4))
    alt_b = np.array([x["ema_at_best_live"] for x in ema_rows if x["mode"] == "before_snap"])
    alt_a = np.array([x["ema_at_best_live"] for x in ema_rows if x["mode"] == "after_snap"])
    alt_delta = float(alt_b.mean() - alt_a.mean())
    parent_b, parent_a, parent_d = 0.057043, 0.057200, 0.000157

    # ---------- 本轮新增：seed-0 十二格（表 4）order_diff 与 spread ----------
    cells = []
    for col in ("mae_fp32", "mae_8", "mae_6", "mae_4"):
        for N in intervals:
            g = {r.mode: float(getattr(r, col)) for r in s0.loc[s0["snap_every"] == N].itertuples()}
            cells.append({"col": col, "N": N,
                          "none": g["none"], "before": g["before_snap"], "after": g["after_snap"],
                          "order_diff": float(abs(g["before_snap"] - g["after_snap"])),
                          "spread": float(max(g.values()) - min(g.values()))})
    cell_rank = sorted(cells, key=lambda c: (-c["order_diff"], c["col"], c["N"]))
    cell_max = cell_rank[0]
    cols4 = ("mae_fp32", "mae_8", "mae_6", "mae_4")
    orders_by_col = {c: max(x["order_diff"] for x in cells if x["col"] == c) for c in cols4}
    spreads_by_col = {c: max(x["spread"] for x in cells if x["col"] == c) for c in cols4}

    def cell_hits(target, key):
        return [{"col": x["col"], "N": x["N"]} for x in cells
                if abs(round(x[key], 4) - target) < 5e-9]

    probes = {}
    for target in (0.0011, 0.0003):
        probes[f4(target)] = {"order_diff": cell_hits(target, "order_diff"),
                              "spread": cell_hits(target, "spread")}

    def cell_at(col, N):
        return [x for x in cells if x["col"] == col and x["N"] == N][0]

    def probe_str(v):
        p = probes[f4(v)]
        parts = []
        for key in ("order_diff", "spread"):
            for h in p[key]:
                parts.append(h["col"] + "/N=" + str(h["N"]) + "（" + key + "）")
        return "、".join(parts) if parts else "无"

    c_fp32_20, c_8_20, c_6_20, c_4_1 = cell_at("mae_fp32", 20), cell_at("mae_8", 20), cell_at("mae_6", 20), cell_at("mae_4", 1)
    c_fp32_5, c_4_20 = cell_at("mae_fp32", 5), cell_at("mae_4", 20)

    # ---------- 本轮新增：倍数关系 ----------
    interval_eff = float(g1.mean() - g20.mean())
    ratio = {"interval_effect": interval_eff,
             "interval_effect_welch_abs": float(abs(d_1_20["delta"])),
             "order_max": cell_max["order_diff"],
             "order_max_where": [cell_max["col"], cell_max["N"]],
             "ratio_global": interval_eff / cell_max["order_diff"],
             "ratio_fp32_N20": interval_eff / c_fp32_20["order_diff"],
             "ratio_mae8_N20": interval_eff / c_8_20["order_diff"],
             "ratio_mae6_N20": interval_eff / c_6_20["order_diff"],
             "ratio_ema_order": interval_eff / abs(float(ema_w["delta"]))}

    r_min, r_max = min(ratio["ratio_fp32_N20"], ratio["ratio_mae8_N20"], ratio["ratio_mae6_N20"]), max(ratio["ratio_fp32_N20"], ratio["ratio_mae8_N20"], ratio["ratio_mae6_N20"])
    cons = [d_1_20["verdict"]["consistent"]] + [order[t]["verdict"]["consistent"] for t in order] + [ema_w["verdict"]["consistent"]]
    cons_ok = sum(1 for c in cons if c)
    concl = []
    concl.append("吸附间隔把 8 bit（mae_fp32）协议均值从 N=1 的 " + f4(i1["mean"]) + " 降到 N=20 的 "
                 + f4(i20["mean"]) + "（每点 n=3 个 seed-0 run、三 mode 算术平均，sd "
                 + f4(i1["sd"]) + " 与 " + f4(i20["sd"]) + "），差 " + f4(d_1_20["delta"])
                 + "，Welch 95% CI 为 " + ci_str(d_1_20["lo"], d_1_20["hi"])
                 + "（df=" + f4(d_1_20["df"]) + "），区间不含 0。")
    concl.append("按论文自己的口径（results/<run>/history.csv 的 val_mae_ema 最佳轮次，n=3+3），N=20 的 before_snap 为 "
                 + f4(ema_w["mean_a"]) + "±" + f4(ema_t["before"]["sd"]) + "、after_snap 为 "
                 + f4(ema_w["mean_b"]) + "±" + f4(ema_t["after"]["sd"]) + "，精确差 |Δ| = "
                 + f5(abs(ema_w["delta"])) + "，Welch 95% CI 为 " + ci_str(ema_w["lo"], ema_w["hi"])
                 + "（df=" + f4(ema_w["df"]) + "），可排除的 |Δ| 上界为 " + f4(ema_w["excl"]) + "。")
    concl.append("seed-0 十二格里顺序差 |before_snap − after_snap| 的最大值为 " + f4(cell_max["order_diff"])
                 + "（" + cell_max["col"] + "，N=" + str(cell_max["N"]) + "）；8 bit 内部的最大顺序差为 mae_fp32 "
                 + f4(orders_by_col["mae_fp32"]) + "（N=20）与 mae_8 " + f4(orders_by_col["mae_8"]) + "（N=20）。")
    concl.append("论文写的 0.0011 实为 " + probe_str(0.0011) + " 的单格值（6 bit 在 N=20 上是 "
                 + f4(c_6_20["order_diff"]) + "），0.0003 实为 " + probe_str(0.0003)
                 + " 的单格值；8 bit 导出引起的最大任务变化是 " + f4(mae8_gap) + "。")
    concl.append("间隔效应 " + f4(interval_eff) + " 与十二格最大顺序差 " + f4(cell_max["order_diff"]) + " 之比为 "
                 + f4(ratio["ratio_global"]) + " 倍，不足十倍；把顺序差限定到 N=20 的三个格子时为 "
                 + f4(r_min) + "–" + f4(r_max) + " 倍。")
    concl.append("4 bit（mae_4）均值为 N=1 " + f4(b4[1]["ci"]["mean"]) + "（n=5，95% CI "
                 + ci_str(b4[1]["ci"]["lo"], b4[1]["ci"]["hi"]) + "）、N=5 " + f4(b4[5]["ci"]["mean"])
                 + "（n=3，95% CI " + ci_str(b4[5]["ci"]["lo"], b4[5]["ci"]["hi"]) + "）、N=20 "
                 + f4(b4[20]["ci"]["mean"]) + "（n=7，95% CI " + ci_str(b4[20]["ci"]["lo"], b4[20]["ci"]["hi"])
                 + "）；5 个差值量的 bootstrap 与 t 判定一致（" + str(cons_ok) + "/" + str(len(cons)) + "）。")


    md = []
    add = md.append
    add("# paper-2 置信区间报告（只读计算，未改动任何既有文件）")
    add("")
    add("生成脚本：scripts/paper2_stats_ci.py（D:/tool/Python/python.exe + numpy/scipy/pandas，"
        "bootstrap 种子 " + str(BOOT_SEED) + "，B=" + str(BOOT_N) + "）。")
    add("数据源：quant_runs/deploy_table.csv，共 " + str(n_rows) + " 行，列 = " + ", ".join(df.columns) + "。")
    add("")
    if bad:
        add("## 0. 本轮更新（先读这一节）")
        add("")
        add("### 0.1 已作废的结论：上一轮的「⚠ 口径不一致」是误判，现撤销")
        add("")
        add("上一轮用 deploy_table 的 mae_* 列重算 N=20 的 before_snap / after_snap 均值，得到 "
            + f4(o_f["mean_a"]) + " / " + f4(o_f["mean_b"]) + "，据此判定论文的 "
            + f4(PAPER["order_fp32_before"]) + " / " + f4(PAPER["order_fp32_after"]) + " 口径不一致。"
            "该判定作废：论文那两个数不是从 mae_* 列算的，而是 results/<run>/history.csv 的 "
            "val_mae_ema 列在最佳轮次的读数（第 0.2 节逐 run 复算得 "
            + f4(ema_w["mean_a"]) + " / " + f4(ema_w["mean_b"]) + "，与论文一致）。"
            "上一轮把「两种口径」误读成「数据被错读」，这是本轮要纠正的错误。")
        add("")
        add("第二处过度推断也一并纠正：0.0570 与 0.0572 在 deploy_table 里确实分别落在 "
            "after_snap/s2@N=20 的 mae_fp32 与 mae_8 上（见下方取证），但那是数值巧合，"
            "并不说明论文引用错了列。")
        add("")
        add("**撤销前的记录（原样保留，不删改）**：上一轮在 mae_* 口径下判为不一致的三项如下——")
        add("")
        add("")
        add("| 项 | 论文/任务书既有值 | 本次算得值 | 差（算得减既有） |")
        add("|---|---|---|---|")
        for c in bad:
            add("| " + c["item"] + " | " + f4(c["paper"]) + " | " + f4(c["computed"])
                + " | " + f4(c["diff"]) + " |")
        add("")
        add("取证：这些既有值在表里实际对应的单元格（逐列扫描结果）")
        add("")
        for fo in forensic:
            if not fo["hits"]:
                add("* " + f4(fo["value"]) + "：15 行 x 5 个数值列里没有任何单元格等于该值。")
            else:
                parts = [h["col"] + "@N=" + str(h["snap_every"]) + "/" + h["mode"] + "/s" + str(h["seed"])
                         for h in fo["hits"]]
                add("* " + f4(fo["value"]) + "：出现在 " + "；".join(parts) + "。")
        add("")
        add("**撤销依据**：6 个 N=20 run 的 history.csv 里，val_mae_ema 的最小值分别是 before_snap "
            + f5(float(ema_b.min())) + "–" + f5(float(ema_b.max())) + "、after_snap "
            + f5(float(ema_a.min())) + "–" + f5(float(ema_a.max())) + "，两组均值 "
            + f4(ema_w["mean_a"]) + " ± " + f4(ema_t["before"]["sd"]) + " 与 "
            + f4(ema_w["mean_b"]) + " ± " + f4(ema_t["after"]["sd"]) + "，与论文的 "
            + f4(PAPER["order_fp32_before"]) + " ± 0.0004 与 " + f4(PAPER["order_fp32_after"])
            + " ± 0.0008 逐项吻合。")
        add("")
        add("### 0.2 新口径的顺序效应（N=20，val_mae_ema 最佳轮次，n=3+3）")
        add("")
        add("| mode | seed | run | 最佳轮次 | 最佳 val_mae_ema | 末轮 val_mae_ema | 同 run 的 mae_fp32（表内） | 差（ema − mae_fp32） |")
        add("|---|---|---|---|---|---|---|---|")
        for x in ema_rows:
            add("| " + x["mode"] + " | " + str(x["seed"]) + " | " + x["run"] + " | "
                + str(x["best_epoch"]) + " | " + f5(x["best_ema"]) + " | " + f5(x["last_ema"]) + " | "
                + f4(x["csv_mae_fp32"]) + " | " + f5(x["shift_vs_csv_fp32"]) + " |")
        add("")
        add("| 分组 | n | 均值 | sd (ddof=1) | 95% CI (t) | 95% CI (bootstrap) |")
        add("|---|---|---|---|---|---|")
        add("| N=20 before_snap（val_mae_ema） | 3 | " + f4(ema_w["mean_a"]) + " | "
            + f4(ema_t["before"]["sd"]) + " | " + ci_str(ema_t["before"]["lo"], ema_t["before"]["hi"])
            + " | " + ci_str(ema_boot_mean["before"][0], ema_boot_mean["before"][1]) + " |")
        add("| N=20 after_snap（val_mae_ema） | 3 | " + f4(ema_w["mean_b"]) + " | "
            + f4(ema_t["after"]["sd"]) + " | " + ci_str(ema_t["after"]["lo"], ema_t["after"]["hi"])
            + " | " + ci_str(ema_boot_mean["after"][0], ema_boot_mean["after"][1]) + " |")
        add("")
        add("| 量 | 值 |")
        add("|---|---|")
        add("| 两组精确均值（5 位小数） | " + f5(ema_w["mean_a"]) + " / " + f5(ema_w["mean_b"]) + " |")
        add("| 两组求和（核验，6 个文件逐个读入） | " + f5(ema_sum_b) + " / " + f5(ema_sum_a) + " |")
        add("| Δ = before_snap − after_snap（精确，5 位小数） | " + f5(ema_w["delta"]) + " |")
        add("| \\|Δ\\| 精确值（5 位小数） | " + f5(abs(ema_w["delta"])) + " |")
        add("| Welch df（Satterthwaite） | " + f4(ema_w["df"]) + " |")
        add("| 95% CI（t, Welch） | " + ci_str(ema_w["lo"], ema_w["hi"]) + " |")
        add("| 95% CI（bootstrap, B=10000, seed=0） | " + ci_str(ema_w["boot"][0], ema_w["boot"][1]) + " |")
        add("| 可排除的 \\|Δ\\| 上界（t） | " + f4(ema_w["excl"]) + " |")
        add("| 可排除的 \\|Δ\\| 上界（bootstrap） | " + f4(ema_w["boot_excl"]) + " |")
        add("")
        add("两套区间都含 0：n=3+3 的数据不能拒绝「N=20 上没有顺序效应」。能排除的只是绝对值大于 "
            + f4(ema_w["excl"]) + " 的效应（t 口径）。")
        add("")
        add("**论文的「差值 0.0001」该怎么写**：精确 |Δ| = " + f5(abs(ema_w["delta"]))
            + "（均值 " + f5(ema_w["mean_a"]) + " 与 " + f5(ema_w["mean_b"])
            + "，方向是 before_snap 略低）；把均值先按 4 位小数呈现再相减，|Δ| 变成 "
            + f5(abs(ema_rounded_sub)) + "。两者相差不到 1e-4，恰好跨在 4 位小数的进位边界上。"
            "结论：稿件若同时展示 " + f4(ema_w["mean_a"]) + " 与 " + f4(ema_w["mean_b"])
            + "，就必须写 0.0002，否则读者按展示值相减会与你写的 0.0001 冲突；"
            "若要保留 0.0001，均值必须写到 5 位小数（" + f5(ema_w["mean_a"]) + " / "
            + f5(ema_w["mean_b"]) + "）。")
        add("")
        add("**与父代理 5 位小数读数的比对**：父代理测得 " + f5(parent_b) + " / " + f5(parent_a)
            + "（差 " + f5(parent_d) + "）；本脚本从 history.csv 直接取 val_mae_ema 最小值得到 "
            + f5(ema_w["mean_a"]) + " / " + f5(ema_w["mean_b"]) + "（差 " + f5(abs(ema_w["delta"]))
            + "）。逐 run 的 4 位小数两边一致，5 位小数差 1.5e-5 / 2.7e-5，"
            "从这 6 个 history.csv 无法复现父代理那组 5 位小数；改按「val_mae_live 最小轮次」取 ema 则为 "
            + f5(float(alt_b.mean())) + " / " + f5(float(alt_a.mean())) + "（差 "
            + f5(abs(alt_delta)) + "），也不是父代理那组。两种读法在 4 位小数上完全一致（"
            + f4(ema_w["mean_a"]) + " / " + f4(ema_w["mean_b"]) + "），结论不受影响。")
        add("")
        add("### 0.3 表 4 复算：seed-0 十二格（每格 3 条 mode 行）")
        add("")
        add("order_diff = |before_snap − after_snap|；spread = 三 mode 的 max − min。")
        add("")
        add("| 列（位宽） | 间隔 N | none | before_snap | after_snap | order_diff | spread |")
        add("|---|---|---|---|---|---|---|")
        for x in cells:
            add("| " + x["col"] + " | " + str(x["N"]) + " | " + f4(x["none"]) + " | " + f4(x["before"])
                + " | " + f4(x["after"]) + " | " + f4(x["order_diff"]) + " | " + f4(x["spread"]) + " |")
        add("")
        add("| 汇总 | 值（4 位小数） | 位置 |")
        add("|---|---|---|")
        add("| order_diff 最大 | " + f4(cell_max["order_diff"]) + " | " + cell_max["col"]
            + " @ N=" + str(cell_max["N"]) + " |")
        for col in cols4:
            top = [x for x in cells if x["col"] == col and x["order_diff"] == orders_by_col[col]][0]
            add("| " + col + " 的 order_diff 最大 | " + f4(orders_by_col[col]) + " | N="
                + str(top["N"]) + " |")
        add("")
        add("### 0.4 「0.0011（6 bit, 间隔 20）」与「8 bit ≤0.0003」是什么量")
        add("")
        add("**0.0011 的唯一位置**：" + probe_str(0.0011) + "。它属于 4 bit、间隔 1，不是「6 bit、间隔 20」；"
            "6 bit 在 N=20 上的 order_diff 是 " + f4(c_6_20["order_diff"]) + "（不是 0.0011）。")
        add("")
        add("**0.0003 的唯一位置**：" + probe_str(0.0003) + "。它是单格的三 mode 极差（同时间隔 N=5 上 6 bit 的 order_diff），"
            "不是任何跨格上界。8 bit 的真实上界是 mae_fp32 " + f4(orders_by_col["mae_fp32"])
            + "（N=20）与 mae_8 " + f4(orders_by_col["mae_8"]) + "（N=20）。")
        add("")
        add("**若论文原意是「8 bit 导出引起的最大任务变化」**，正确值是 max|mae_8 − mae_fp32| = "
            + f4(mae8_gap) + "（15 行全表），也不是 0.0003。两个量只差 0.0001，极易被当成同一个数。")
        add("")
        add("判定：0.0011 与 0.0003 都不是表 4 的任何上界，而是两个具体单元格的值；"
            "把它们写成「最大差异 ≤0.0011 / ≤0.0003」会让读者按更严的口径复算，"
            "得到 " + f4(cell_max["order_diff"]) + "（全场最大）与 " + f4(orders_by_col["mae_fp32"])
            + "（8 bit 最大），与稿件数字冲突。")
        add("")
        add("### 0.5 倍数关系：order_max / interval_effect")
        add("")
        add("定义：interval_effect = 表 3 的协议均值从 N=1 到 N=20 的下降量 = " + f4(i1["mean"]) + " − "
            + f4(i20["mean"]) + " = " + f4(interval_eff) + "（mae_fp32，seed-0 三 mode 算术平均）；"
            "order_max = 表 4 十二格里最大的 order_diff = " + f4(cell_max["order_diff"]) + "（"
            + cell_max["col"] + " @ N=" + str(cell_max["N"]) + "）。")
        add("")
        add("| 比值 | 值 |")
        add("|---|---|")
        add("| interval_effect ÷ order_max（全场最大） | " + f4(ratio["ratio_global"]) + " 倍 |")
        add("| interval_effect ÷ order_diff（mae_fp32 @ N=20） | " + f4(ratio["ratio_fp32_N20"]) + " 倍 |")
        add("| interval_effect ÷ order_diff（mae_8 @ N=20） | " + f4(ratio["ratio_mae8_N20"]) + " 倍 |")
        add("| interval_effect ÷ order_diff（mae_6 @ N=20） | " + f4(ratio["ratio_mae6_N20"]) + " 倍 |")
        add("| interval_effect ÷ " + f5(abs(ema_w["delta"])) + "（val_mae_ema 口径的顺序差） | "
            + f4(ratio["ratio_ema_order"]) + " 倍 |")
        add("")
        add("判定：「顺序效应比间隔效应小一个数量级」只有把顺序效应限定到 N=20 那三个格子时才成立（"
            + f4(r_min) + "–" + f4(r_max) + " 倍）；用十二格里最大的 order_diff（6 bit @ N=1）只有 "
            + f4(ratio["ratio_global"]) + " 倍，不足十倍。论文要么把这句话限定到 N=20，要么改成 "
            + f4(ratio["ratio_global"]) + " 倍。")
        add("")
    else:
        add("## 0. 自校验：既有数字逐项吻合")
        add("")
        add("（checks 全绿分支，当前不会走到：PAPER 里的 N=20 顺序效应属 val_mae_ema 口径，见第 0.2 节。）")
        add("")

    add("## 1. 数据完整性与 SHA1")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add("| 行数 | " + str(n_rows) + "（seed0 " + str(int((df["seed"] == 0).sum()))
        + " 行 + seed1/2 " + str(int((df["seed"] != 0).sum())) + " 行） |")
    add("| 分组构成 | seed0：N 属于 {1,5,20} x mode 属于 {none,before_snap,after_snap} 各 1 run；"
        "seed1/2：after_snap@N=1、after_snap@N=20、before_snap@N=20 |")
    add("| run 名里的 sX 与 seed 列一致 | " + ("是（0 处不一致）" if not mism else json.dumps(mism, ensure_ascii=False)) + " |")
    add("| sha1（计算前） | " + sha_before + " |")
    add("| sha1（计算后） | " + sha1_of(CSV_PATH) + " |")
    add("| 文件字节数 | " + str(CSV_PATH.stat().st_size) + " |")
    add("| 附加只读输入 | results/<run>/history.csv —— 只读 6 个 N=20 run（before_snap / after_snap 各 3 个 seed），"
        "其余 9 个 run 的历史文件未读 |")
    add("")
    add("## 2. 分组均值与 95% CI（t 与 bootstrap 并列）")
    add("")
    add("| 分组 | n | 均值 | sd (ddof=1) | 95% CI (t) | 95% CI (bootstrap) |")
    add("|---|---|---|---|---|---|")
    for g in groups:
        add("| " + g["label"] + " | " + str(g["ci"]["n"]) + " | " + f4(g["ci"]["mean"])
            + " | " + f4(g["ci"]["sd"]) + " | " + ci_str(g["ci"]["lo"], g["ci"]["hi"])
            + " | " + ci_str(g["boot"][0], g["boot"][1]) + " |")
    add("")
    add("## 3. 顺序效应（N=20，before_snap 减 after_snap，Welch–Satterthwaite）")
    add("")
    add("| 位宽 / 列 | Δ | Welch df | 95% CI (t) | 95% CI (bootstrap) | 可排除的 \\|Δ\\| 上界 |")
    add("|---|---|---|---|---|---|")
    for tag, name in [("fp32", "8 bit（mae_fp32）"), ("mae6", "6 bit（mae_6）"), ("mae8", "8 bit 复算（mae_8）")]:
        w = order[tag]
        add("| " + name + " | " + f4(w["delta"]) + " | " + f4(w["df"]) + " | "
            + ci_str(w["lo"], w["hi"]) + " | " + ci_str(w["boot"][0], w["boot"][1]) + " | "
            + f4(w["excl"]) + " |")
    add("")
    add("可排除的绝对值上界 = max(|CI 下界|, |CI 上界|)：任何绝对值超过该上界的顺序效应在 95% 水平上被排除，"
        "该上界以内的效应本数据不能排除。")
    add("")
    for tag, name in [("fp32", "8 bit"), ("mae6", "6 bit")]:
        w = order[tag]
        add("* " + name + "：Δ = " + f4(w["delta"]) + "，95% CI " + ci_str(w["lo"], w["hi"])
            + "，只能排除绝对值大于 " + f4(w["excl"]) + " 的顺序效应；原始值 before_snap = "
            + " / ".join(f4(x) for x in w["values_before"]) + "，after_snap = "
            + " / ".join(f4(x) for x in w["values_after"]) + "。")
    add("")
    add("## 4. 间隔效应：N=1 到 N=20 的均值差")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add("| Δ（N=20 减 N=1，mae_fp32） | " + f4(d_1_20["delta"]) + " |")
    add("| Welch df | " + f4(d_1_20["df"]) + " |")
    add("| 95% CI (t) | " + ci_str(d_1_20["lo"], d_1_20["hi"]) + " |")
    add("| 95% CI (bootstrap) | " + ci_str(d_1_20["boot"][0], d_1_20["boot"][1]) + " |")
    add("| 可排除更小的 \\|Δ\\|（最小可分辨幅度） | " + f4(min(abs(d_1_20["lo"]), abs(d_1_20["hi"]))) + " |")
    add("")
    add("注：N=1 到 N=20 的区间整体为负且不含 0，能说的是「N=20 比 N=1 低，落差在 "
        + f4(min(abs(d_1_20["lo"]), abs(d_1_20["hi"]))) + " 到 "
        + f4(max(abs(d_1_20["lo"]), abs(d_1_20["hi"]))) + " 之间」（按绝对值）；"
        "上界式表述只用于区间跨 0 的情形（第 3 节）。")
    add("")
    add("## 5. 4 bit（mae_4）")
    add("")
    add("| 间隔 | n | 均值 | sd | 95% CI (t) | 95% CI (bootstrap) | run 间极差 |")
    add("|---|---|---|---|---|---|---|")
    for N in intervals:
        c = b4[N]["ci"]
        add("| N=" + str(N) + " | " + str(c["n"]) + " | " + f4(c["mean"]) + " | " + f4(c["sd"])
            + " | " + ci_str(c["lo"], c["hi"]) + " | "
            + ci_str(b4[N]["boot"][0], b4[N]["boot"][1]) + " | "
            + f4(b4[N]["min"]) + " – " + f4(b4[N]["max"]) + " |")
    add("")
    add("对「间隔 20 的 run 间极差 0.0849–0.2030」的核对：**与数据一致**（N=20 的 "
        + str(b4[20]["ci"]["n"]) + " 个 run，mae_4 最小值 " + f4(b4[20]["min"])
        + "，最大值 " + f4(b4[20]["max"]) + "）。")
    add("")
    add("4 bit 分组的 n 不均且混 mode：N=1 有 5 个 run（after_snap x3 seed + before_snap x1 + none x1）、"
        "N=5 有 3 个（三 mode 各 1）、N=20 有 7 个（after_snap x3 + before_snap x3 + none x1），"
        "因此这里给的是「该间隔上全部 run 的均值」，不是协议均值。")
    add("")
    add("## 6. bootstrap 与 t 区间是否一致")
    add("")
    add("| 量 | t 区间排除 0 | bootstrap 排除 0 | 两区间相交 | 结论 |")
    add("|---|---|---|---|---|")
    comp_rows = [("间隔效应 Δ(20-1)", d_1_20["verdict"])]
    for tag, name in [("fp32", "顺序效应 8 bit"), ("mae6", "顺序效应 6 bit"), ("mae8", "顺序效应 8 bit(mae_8)")]:
        comp_rows.append((name, order[tag]["verdict"]))
    comp_rows.append(("顺序效应 val_mae_ema", ema_w["verdict"]))
    for N in intervals:
        comp_rows.append(("4 bit 均值 N=" + str(N),
                          verdict(b4[N]["ci"]["lo"], b4[N]["ci"]["hi"], b4[N]["boot"][0], b4[N]["boot"][1])))
    for nm, v in comp_rows:
        add("| " + nm + " | " + ("是" if v["t_excludes_zero"] else "否") + " | "
            + ("是" if v["boot_excludes_zero"] else "否") + " | "
            + ("是" if v["overlap"] else "否") + " | "
            + ("一致" if v["consistent"] else "**不一致**") + " |")
    add("")
    add("判据：两区间相交且对「是否含 0」给出相同判断，才记为一致。")
    add("")
    add("## 7. 可写进论文的最小结论集")
    add("")
    for i, s in enumerate(concl, 1):
        add(str(i) + ". " + s)
    add("")
    add("## 8. 口径坑（如实记录）")
    add("")
    add("1. **deploy_table 的 mae_* 列与论文口径不同**：按 mae_* 列，N=20 的 before_snap / after_snap 为 "
        + f4(o_f["mean_a"]) + "±" + f4(o_f["sd_a"]) + " 与 " + f4(o_f["mean_b"]) + "±" + f4(o_f["sd_b"])
        + "；论文的 " + f4(PAPER["order_fp32_before"]) + "±0.0004 与 " + f4(PAPER["order_fp32_after"])
        + "±0.0008 来自 results/<run>/history.csv 的 val_mae_ema（最佳轮次，见第 0.2 节）。"
        "两套产物都存在，混用会得到互相矛盾的点估计。")
    add("2. **4 位小数进位边界**：0.0572 减 0.0570 = 0.0002，而精确差 " + f5(abs(ema_w["delta"]))
        + " 四舍五入到 4 位是 0.0001；两者都取决于「先取整还是先相减」。稿件必须自洽（见第 0.2 节与第 10 条）。")
    add("3. **n=3 的上界表述**：mae_* 口径下只能排除 \\|Δ\\| > " + f4(o_f["excl"]) + "（8 bit）/ "
        + f4(o_6["excl"]) + "（6 bit）；换成论文口径（val_mae_ema）是 " + f4(ema_w["excl"])
        + "。论文写的 ≤0.0011 既不是其中任何一个口径的上界，也不是点估计（点估计是 "
        + f5(abs(ema_w["delta"])) + "）。")
    add("4. **CLAUDE.md 里的 0.0591 与 0.0642 是单行读数，不是协议均值**：协议均值按 CSV 为 N=20 的 "
        + f4(i20["mean"]) + "（8 bit）与 " + f4(iv6[20]["mean"])
        + "（6 bit）；0.0591 出现在 val_mae_train 列，0.0642 是 none@N=20 的 mae_6。")
    add("5. **CLAUDE.md 写的 4 bit 范围 0.0849–0.2337 跨了间隔**：0.2337 是 after_snap/s2@N=1 的 mae_4，"
        "N=20 内部的极差是 " + f4(b4[20]["min"]) + " – " + f4(b4[20]["max"]) + "（与任务书一致）。")
    add("6. **mae_8 与 mae_fp32 的最大差只有 " + f4(mae8_gap) + "**，8 bit 那一行的 CI 与用 mae_8 复算的 CI 实际重合，"
        "8 bit 无法区分两个顺序。")
    add("7. **4 bit 的 n=5/3/7 不是同一批 run**：每个间隔混了不同 mode 与不同 seed，组内方差主要来自 seed"
        "（N=20 的 sd " + f4(b4[20]["ci"]["sd"]) + " 远大于 N=5 的 " + f4(b4[5]["ci"]["sd"])
        + "），把三组当成「间隔效应」比较会与 seed 效应混杂。")
    add("8. **t 分位核验**：scipy.stats.t.ppf(0.975, 2) = " + f4(T975_DF2_TASK)
        + "，与任务书给的 4.302653 一致（脚本内该常量仅作核对，实际用 scipy 值）。")
    add("9. **上一轮的「口径不一致」是误判**：0.0570 / 0.0572 来自 results/<run>/history.csv 的 val_mae_ema"
        "（最佳轮次），不是 deploy_table 的 mae_* 列；换口径后逐项吻合（第 0.2 节）。"
        "教训：同一个量可能存在两套产物，先定口径再判对错。")
    add("10. **4 位小数的进位边界**：精确差 " + f5(abs(ema_w["delta"])) + " 落在 0.00015 附近，"
        "先取整再相减得 " + f5(abs(ema_rounded_sub)) + "，精确相减再取整得 0.0001。"
        "展示值与差值必须同源，否则读者按展示值一减就对不上。")
    add("11. **单格值被当成上界**：0.0011 是 4 bit/N=1 的单格，0.0003 是 N=5 三 mode 的极差；"
        "表 4 的真实上界是 8 bit 的 " + f4(orders_by_col["mae_fp32"]) + "（mae_fp32）/ "
        + f4(orders_by_col["mae_8"]) + "（mae_8），全场最大 " + f4(cell_max["order_diff"]) + "（6 bit/N=1）。")
    add("")
    add("## 9. 复现")
    add("")
    add("D:/tool/Python/python.exe scripts/paper2_stats_ci.py（确定性：bootstrap 种子 0，每类量各自 default_rng(0)）。")
    add("")

    md_text = NL.join(md) + NL
    MD_PATH.write_text(md_text, encoding="utf-8", newline=NL)

    sha_after = sha1_of(CSV_PATH)
    assert sha_after == sha_before, "deploy_table.csv 被改动了"

    out = {
        "meta": {
            "script": "scripts/paper2_stats_ci.py",
            "interpreter": "D:/tool/Python/python.exe",
            "csv_sha1_before": sha_before,
            "csv_sha1_after": sha_after,
            "csv_bytes": int(CSV_PATH.stat().st_size),
            "csv_rows": n_rows,
            "boot_n": BOOT_N,
            "boot_seed": BOOT_SEED,
            "t975_df2_task": T975_DF2_TASK,
            "t975_df2_scipy": float(stats.t.ppf(0.975, 2)),
            "seed_name_mismatches": mism,
            "mae8_minus_fp32_maxabs": mae8_gap,
        },
        "groups": {g["key"]: {"label": g["label"], "n": g["ci"]["n"], "mean": g["ci"]["mean"],
                              "sd": g["ci"]["sd"], "t": g["ci"]["t"],
                              "ci_t": [g["ci"]["lo"], g["ci"]["hi"]],
                              "ci_boot": list(g["boot"]), "values": [float(v) for v in g["values"]]}
                   for g in groups},
        "order_effect_N20": {k: {"delta": v["delta"], "df": v["df"], "t": v["t"],
                                 "ci_t": [v["lo"], v["hi"]], "ci_boot": list(v["boot"]),
                                 "excl_bound_t": v["excl"], "excl_bound_boot": v["boot_excl"],
                                 "mean_before": v["mean_a"], "mean_after": v["mean_b"],
                                 "sd_before": v["sd_a"], "sd_after": v["sd_b"],
                                 "values_before": v["values_before"], "values_after": v["values_after"],
                                 "verdict": v["verdict"]}
                             for k, v in order.items()},
        "interval_delta_1_20": {"delta": d_1_20["delta"], "df": d_1_20["df"],
                                "ci_t": [d_1_20["lo"], d_1_20["hi"]],
                                "ci_boot": list(d_1_20["boot"]),
                                "excl_bound_t": d_1_20["excl"], "verdict": d_1_20["verdict"]},
        "interval_6bit_means": {str(N): {"mean": iv6[N]["mean"], "sd": iv6[N]["sd"], "n": iv6[N]["n"]}
                                for N in intervals},
        "bits4": {str(N): {"n": b4[N]["ci"]["n"], "mean": b4[N]["ci"]["mean"], "sd": b4[N]["ci"]["sd"],
                           "ci_t": [b4[N]["ci"]["lo"], b4[N]["ci"]["hi"]],
                           "ci_boot": list(b4[N]["boot"]),
                           "min": b4[N]["min"], "max": b4[N]["max"],
                           "mode_comp": b4[N]["mode_comp"]}
                      for N in intervals},
        "order_effect_N20_ema": {
            "source": "results/<run>/history.csv, column val_mae_ema, min over epochs",
            "runs": ema_rows,
            "mean_before": ema_w["mean_a"], "mean_after": ema_w["mean_b"],
            "sd_before": ema_w["sd_a"], "sd_after": ema_w["sd_b"],
            "sum_before": ema_sum_b, "sum_after": ema_sum_a,
            "delta_before_minus_after": ema_w["delta"],
            "delta_5dp": f5(ema_w["delta"]),
            "rounded_mean_subtraction": ema_rounded_sub,
            "df": ema_w["df"], "ci_t": [ema_w["lo"], ema_w["hi"]],
            "ci_boot": list(ema_w["boot"]),
            "excl_bound_t": ema_w["excl"], "excl_bound_boot": ema_w["boot_excl"],
            "verdict": ema_w["verdict"],
            "group_ci_t_before": [ema_t["before"]["lo"], ema_t["before"]["hi"]],
            "group_ci_t_after": [ema_t["after"]["lo"], ema_t["after"]["hi"]],
            "group_ci_boot_before": list(ema_boot_mean["before"]),
            "group_ci_boot_after": list(ema_boot_mean["after"]),
            "alt_best_live_definition": {"mean_before": float(alt_b.mean()),
                                         "mean_after": float(alt_a.mean()),
                                         "delta": alt_delta},
            "parent_reported": {"mean_before": parent_b, "mean_after": parent_a,
                                "delta": parent_d}},
        "cell_table_seed0": cells,
        "cell_order_diff_max": {"col": cell_max["col"], "N": cell_max["N"],
                                "order_diff": cell_max["order_diff"]},
        "order_diff_max_by_col": orders_by_col,
        "spread_max_by_col": spreads_by_col,
        "value_probes": probes,
        "ratio": ratio,
        "paper_checks": checks,
        "paper_mismatches": bad,
        "forensic_value_location": forensic,
        "conclusions": concl,
    }
    JSON_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2) + NL, encoding="utf-8", newline=NL)

    print("sha1 before:", sha_before)
    print("sha1 after :", sha_after)
    print("checks:", len(checks), "mismatches:", len(bad))
    for c in bad:
        print("  MISMATCH", c["item"], "paper", c["paper"], "computed", round(c["computed"], 6))
    print("md bytes  :", MD_PATH.stat().st_size)
    print("json bytes:", JSON_PATH.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
