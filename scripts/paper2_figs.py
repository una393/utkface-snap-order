"""篇 2 的两张图。数据全部读自 quant_runs/，不手填任何数。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# 按印刷尺寸出图：宽 5.9 in ≈ 14.5 cm（缩放 0.97），字号 7.4–8.0 pt → 等效 ≥7.1 pt
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.6, "axes.labelsize": 7.5,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 7.4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "lines.linewidth": 1.1, "lines.markersize": 3.4, "patch.linewidth": 0.6,
})

FREQS = [1, 5, 20]
COL = {"fp32": "#333333", "8": "#4a7ba7", "6": "#d08a2c", "4": "#b33b3b"}
MODES = ["none", "before_snap", "after_snap"]
STE_TABLE = "quant_runs/ste_deploy_table.csv"


def main_rows() -> list[dict]:
    """主配置 run（γ = 0.99 且 split_seed = seed）。deploy_table.csv 是退役的 15 行旧表。"""
    rows = [r for r in csv.DictReader(open("quant_runs/deploy_table_v2.csv"))
            if float(r["ema_decay"]) == 0.99 and int(r["split_seed"]) == int(r["seed"])]
    if len(rows) != 23:
        raise SystemExit(f"deploy_table_v2.csv: expected 23 main-config rows, got {len(rows)}")
    return rows


def fig1_deploy_mae(rows: list[dict] | None = None) -> Path:
    """图 1：(a) 部署 MAE vs 吸附间隔（seed 0，三协议均值 + 协议间范围）；(b) 4bit 逐 run。

    (a) 最右端是 N = inf（训练期完全不投影），只有 none / after_snap 两臂
    （before_snap 与 after_snap 是同一算法），取 deploy_table_v2.csv 的 snap_every=0 两行。
    """
    if rows is None:
        rows = main_rows()
    inf = [r for r in csv.DictReader(open("quant_runs/deploy_table_v2.csv"))
           if int(r["snap_every"]) == 0 and r["seed"] == "0"]
    if len(inf) != 2:
        raise SystemExit(f"deploy_table_v2.csv: expected 2 rows at snap_every=0, got {len(inf)}")
    out = Path("drafts/figs")
    out.mkdir(parents=True, exist_ok=True)

    # ---- 图 1：(a) MAE vs 吸附间隔（seed0，三模式均值±极差）；(b) 4bit 逐 run ----
    fig, ax = plt.subplots(1, 2, figsize=(5.9, 2.19))   # 14.5 cm 宽下缩放 0.97
    X_INF = 30.0                    # inf 的占位横坐标：与 20 之间画轴断口，该段非等比
    X = FREQS + [X_INF]
    yy = []                         # 实际画出（不含越界的 4bit@inf）的 lo/hi，用来定纵轴
    for col, key in (("fp32", "mae_fp32"), ("8", "mae_8"), ("6", "mae_6"), ("4", "mae_4")):
        mean, lo, hi = [], [], []
        for f in FREQS:
            v = [float(r[key]) for r in rows if int(r["snap_every"]) == f and r["seed"] == "0"]
            mean.append(np.mean(v)); lo.append(np.min(v)); hi.append(np.max(v))
        if col != "4":              # 4bit 在 inf 均值 0.129，越出纵轴，读数写进注记，曲线止于 N=20
            v_inf = [float(r[key]) for r in inf]
            mean.append(np.mean(v_inf)); lo.append(np.min(v_inf)); hi.append(np.max(v_inf))
        yy += lo + hi
        ax[0].plot(X[:len(mean)], mean, "o-", color=COL[col], label=f"{col.upper() if col == 'fp32' else col + '-bit'}")
        ax[0].fill_between(X[:len(mean)], lo, hi, color=COL[col], alpha=0.12, linewidth=0)
    span = max(yy) - min(yy)
    ax[0].set_ylim(min(yy) - 0.05 * span, max(yy) + 0.05 * span)   # 同 matplotlib 默认 5% 边距
    ax[0].set_xticks(X); ax[0].set_xticklabels(["1", "5", "20", "inf (off)"])
    ax[0].set_xlim(0.0, 36.0)
    ax[0].set_xlabel("snap interval N (steps)")
    XB = 25.0 / 36.0                # 断口落在 20 与 inf 之间（轴分数）
    for y0 in (0.0, 1.0):           # 上下轴线各一对平行斜杠，表示横轴在此断开
        for dx in (-0.009, 0.009):
            ax[0].plot([XB + dx - 0.008, XB + dx + 0.008], [y0 - 0.022, y0 + 0.022],
                       transform=ax[0].transAxes, color="black", lw=0.8, clip_on=False, zorder=6)
    ax[0].text(0.17, 0.76, "N = inf: 2 arms only\n(before = after)\n4-bit@inf 0.129, off",
               transform=ax[0].transAxes, ha="left", va="top", fontsize=7.4,
               color="#3b3b3b", linespacing=1.35)
    ax[0].set_ylabel("task MAE after deployment\nquantization")
    ax[0].set_title("(a) interval is the knob\n(seed 0, 3 arms; N = inf: 2 arms)")
    ste = {r["arm"]: r for r in csv.DictReader(open(STE_TABLE))}["ste_mean"]
    ste_lo, ste_hi = float(ste["mae_8"]), float(ste["mae_fp32"])
    ax[0].axhspan(ste_lo, ste_hi, color="#3f7a4a", alpha=0.20, lw=0, zorder=0)
    for v in (ste_lo, ste_hi):
        ax[0].axhline(v, color="#3f7a4a", lw=0.7, ls=(0, (3, 2)), alpha=0.85, zorder=1)
    ax[0].text(6.0, ste_lo + 0.0005,
               f"STE (no overwrite)\n{ste_lo:.4f} / {ste_hi:.4f}",
               fontsize=6.6, color="#2f5c37", ha="left", va="bottom",
               linespacing=1.3, zorder=6)
    ax[0].legend(fontsize=7.4, loc="center right"); ax[0].grid(alpha=0.25, lw=0.4)

    for s, mk in (("0", "o"), ("1", "s"), ("2", "^")):
        xs, ys = [], []
        for f in FREQS:
            v = [float(r["mae_4"]) for r in rows if int(r["snap_every"]) == f and r["seed"] == s]
            if v:
                xs.append(f); ys.append(np.mean(v))
        ax[1].scatter(xs, ys, marker=mk, s=20, label=f"seed {s}", zorder=3)
    ax[1].set_xticks(FREQS); ax[1].set_xlabel("snap interval N (steps)")
    ax[1].set_ylabel("4-bit task MAE")
    ax[1].set_title("(b) at 4 bits, run-to-run\nspread dominates the interval")
    ax[1].legend(fontsize=7.4); ax[1].grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(out / "fig2.png", dpi=300)
    plt.close(fig)
    return out / "fig2.png"


def main() -> None:
    raise SystemExit(
        "superseded: figure 3(b) and figure 5(a) are produced by scripts/paper2_figs_r20.py "
        "from quant_runs/deploy_table_v4.csv (fixed split, 8 initial seeds). Running this script "
        "would overwrite them with the retired 21-run basis.")
    facts = json.load(open("quant_runs/paper2_facts.json", encoding="utf-8"))
    out = Path("drafts/figs")
    out.mkdir(parents=True, exist_ok=True)

    fig1_deploy_mae(rows)

    # ---- 图 2：权重空间有差异，函数空间没有 ----
    fig, ax = plt.subplots(1, 2, figsize=(5.9, 2.19))   # 14.5 cm 宽下缩放 0.97
    w = 0.26
    for i, m in enumerate(MODES):
        vals = []
        for f in FREQS:
            v = [float(r["mae_fp32"]) for r in rows
                 if int(r["snap_every"]) == f and r["mode"] == m and r["seed"] == "0"]
            vals.append(v[0] if v else np.nan)
        xs = np.arange(len(FREQS))
        ax[0].bar(xs + (i - 1) * w, vals, width=w,
                  label=m, color=["#888888", "#7aa6c2", "#c98b8b"][i])
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax[0].text(x + (i - 1) * w, v + 0.0004, f"{v:.4f}", ha="center", va="bottom",
                           rotation=90, fontsize=7.4)
    ax[0].set_xticks(np.arange(len(FREQS)))
    ax[0].set_xticklabels([f"N={f}" for f in FREQS]); ax[0].set_xlabel("snap interval")
    ax[0].set_ylabel("FP32 task MAE")
    ax[0].set_ylim(0.05, 0.089)    # 上界抬高，给竖排数值标注留位置
    ax[0].set_title("(a) function space: ordering is flat")
    ax[0].legend(fontsize=7.4, loc="upper center", ncol=3, handlelength=1.2,
                 columnspacing=1.0, handletextpad=0.4)
    ax[0].grid(alpha=0.25, axis="y", lw=0.4)

    ds = facts["distances"]
    groups = [("shadow vs live\n(same run)", [x["rel_l2"] for x in ds["shadow_vs_live"]]),
              ("before vs after\n(same seed)", [x["rel_l2"] for x in ds["before_vs_after"]]),
              ("same config\ndifferent seed", [x["rel_l2"] for x in ds["across_seed"]])]
    for i, (name, vals) in enumerate(groups):
        ax[1].scatter([i] * len(vals), vals, s=34, color=["#4a7ba7", "#d08a2c", "#8b5a9e"][i], zorder=3)
        ax[1].plot([i - 0.22, i + 0.22], [np.median(vals)] * 2, color="black", lw=1.6, zorder=4)
    ax[1].set_yscale("log"); ax[1].set_xticks(range(3))
    ax[1].set_xticklabels([g[0] for g in groups], fontsize=7.4)
    ax[1].set_ylabel("relative L2 between\ndeployed weight vectors")
    ax[1].set_title("(b) weight space: ordering moves 4.3-9.1 %")
    ax[1].grid(alpha=0.25, axis="y", lw=0.4)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(out / "figS6.png", dpi=300)
    plt.close(fig)
    print("wrote", list(out.glob("*.png")))


if __name__ == "__main__":
    main()
