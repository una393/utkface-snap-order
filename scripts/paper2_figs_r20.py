"""第二十轮重画：图 3(b) 与图 5(a) 改读 deploy_table_v4.csv（固定划分 split_seed = 0、8 个初始 seed）。

图 3(a) 与图 5(b) 与旧版逐像素同源（同一份代码、同一份 v2 数据），只有被重画的面板取新口径。
"""
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
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
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
V4 = "quant_runs/deploy_table_v4.csv"
SEQ = [("none", 0), ("none", 1), ("none", 20), ("after_snap", 20), ("before_snap", 20)]


def main_rows() -> list[dict]:
    rows = [r for r in csv.DictReader(open("quant_runs/deploy_table_v2.csv"))
            if float(r["ema_decay"]) == 0.99 and int(r["split_seed"]) == int(r["seed"])]
    if len(rows) != 23:
        raise SystemExit(f"deploy_table_v2.csv: expected 23 main-config rows, got {len(rows)}")
    return rows


def seq_rows() -> dict[tuple[str, int], list[dict]]:
    """固定划分（split_seed = 0、γ = 0.99）的五条序列，各 8 个初始 seed。"""
    rows = [r for r in csv.DictReader(open(V4, encoding="utf-8"))
            if r["split_seed"].strip() == "0" and abs(float(r["ema_decay"]) - 0.99) < 1e-9]
    out = {k: sorted((r for r in rows if (r["mode"], int(r["snap_every"])) == k),
                     key=lambda r: int(r["seed"])) for k in SEQ}
    got = {k: [int(r["seed"]) for r in v] for k, v in out.items()}
    if any(v != list(range(8)) for v in got.values()):
        raise SystemExit(f"v4 sequences are not seeds 0..7: {got}")
    return out


def stat(g: list[dict], key: str) -> tuple[float, float]:
    v = np.array([float(r[key]) for r in g])
    return float(v.mean()), float(v.std(ddof=1))


def fig1_deploy_mae(rows: list[dict], seq: dict) -> Path:
    inf = [r for r in csv.DictReader(open("quant_runs/deploy_table_v2.csv"))
           if int(r["snap_every"]) == 0 and r["seed"] == "0"]
    if len(inf) != 2:
        raise SystemExit(f"deploy_table_v2.csv: expected 2 rows at snap_every=0, got {len(inf)}")
    out = Path("drafts/figs")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(5.9, 2.19))
    X_INF = 30.0
    X = FREQS + [X_INF]
    yy = []
    for col, key in (("fp32", "mae_fp32"), ("8", "mae_8"), ("6", "mae_6"), ("4", "mae_4")):
        mean, lo, hi = [], [], []
        for f in FREQS:
            v = [float(r[key]) for r in rows if int(r["snap_every"]) == f and r["seed"] == "0"]
            mean.append(np.mean(v)); lo.append(np.min(v)); hi.append(np.max(v))
        if col != "4":
            v_inf = [float(r[key]) for r in inf]
            mean.append(np.mean(v_inf)); lo.append(np.min(v_inf)); hi.append(np.max(v_inf))
        yy += lo + hi
        ax[0].plot(X[:len(mean)], mean, "o-", color=COL[col], label=f"{col.upper() if col == 'fp32' else col + '-bit'}")
        ax[0].fill_between(X[:len(mean)], lo, hi, color=COL[col], alpha=0.12, linewidth=0)
    span = max(yy) - min(yy)
    ax[0].set_ylim(min(yy) - 0.05 * span, max(yy) + 0.05 * span)
    ax[0].set_xticks(X); ax[0].set_xticklabels(["1", "5", "20", "inf (off)"])
    ax[0].set_xlim(0.0, 36.0)
    ax[0].set_xlabel("snap interval N (steps)")
    XB = 25.0 / 36.0
    for y0 in (0.0, 1.0):
        for dx in (-0.009, 0.009):
            ax[0].plot([XB + dx - 0.008, XB + dx + 0.008], [y0 - 0.022, y0 + 0.022],
                       transform=ax[0].transAxes, color="black", lw=0.8, clip_on=False, zorder=6)
    ax[0].text(0.17, 0.76, "N = inf: 2 arms only\n(before = after)\n4-bit at N = inf: see (b)",
               transform=ax[0].transAxes, ha="left", va="top", fontsize=7.4,
               color="#3b3b3b", linespacing=1.35)
    ax[0].set_ylabel("task MAE after deployment\nquantization")
    ax[0].set_title("(a) interval is the knob\n(seed 0, 3 protocols; N = inf: 2 protocols)")
    ste = {r["arm"]: r for r in csv.DictReader(open(STE_TABLE))}["ste_mean"]
    ste_lo, ste_hi = float(ste["mae_8"]), float(ste["mae_fp32"])
    ax[0].axhspan(ste_lo, ste_hi, color="#3f7a4a", alpha=0.20, lw=0, zorder=0)
    for v in (ste_lo, ste_hi):
        ax[0].axhline(v, color="#3f7a4a", lw=0.7, ls=(0, (3, 2)), alpha=0.85, zorder=1)
    ax[0].text(6.0, ste_lo + 0.0005, f"STE (no overwrite)\n{ste_lo:.4f} / {ste_hi:.4f}",
               fontsize=6.6, color="#2f5c37", ha="left", va="bottom", linespacing=1.3, zorder=6)
    ax[0].legend(fontsize=7.4, loc="center right"); ax[0].grid(alpha=0.25, lw=0.4)

    # ---- (b) 4 bit：固定划分、8 个初始 seed 的五条序列（含 N = inf 端点）----
    rng = np.random.default_rng(0)
    spec = [(("none", 1), 0.0, "o", "#b33b3b", True),
            (("none", 20), 1.0, "o", "#b33b3b", True),
            (("none", 0), 2.0, "o", "#b33b3b", True),
            (("after_snap", 20), 1.0 + 0.17, "s", "#4a7ba7", False),
            (("before_snap", 20), 1.0 - 0.17, "^", "#d08a2c", False)]
    for k, x, mk, c, filled in spec:
        v = np.array([float(r["mae_4"]) for r in seq[k]])
        jit = rng.uniform(-0.045, 0.045, size=v.size)
        ax[1].scatter(x + jit, v, marker=mk, s=10, facecolor=c if filled else "white",
                      edgecolor=c, lw=(0.0 if filled else 0.8), alpha=0.9, zorder=3)
        ax[1].plot([x - 0.10, x + 0.10], [v.mean()] * 2, color=c, lw=1.5, zorder=4)
        ax[1].errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="none", ecolor=c,
                       elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)
    ax[1].set_xticks([0.0, 1.0, 2.0])
    ax[1].set_xticklabels(["N = 1", "N = 20", "inf (off)"], fontsize=7.2)
    ax[1].set_xlim(-0.42, 2.42)
    h = [Line2D([], [], marker="o", ls="", color="#b33b3b", label="live"),
         Line2D([], [], marker="s", ls="", mfc="white", mec="#4a7ba7", label="ema_after @ 20"),
         Line2D([], [], marker="^", ls="", mfc="white", mec="#d08a2c", label="ema_before @ 20")]
    ax[1].legend(handles=h, loc="upper right", fontsize=6.9, frameon=False,
                 handletextpad=0.3, borderpad=0.1, labelspacing=0.25)
    ax[1].set_xlabel("snap interval N (steps); 8 seeds, fixed split")
    ax[1].set_ylabel("4-bit task MAE")
    ax[1].set_title("(b) at 4 bits, seed spread dominates;\nN = 1 is the worst, not the best")
    ax[1].grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    p = out / "fig2.png"
    fig.savefig(p, dpi=660)
    plt.close(fig)
    return p


def fig2_weight_vs_function(seq: dict) -> Path:
    out = Path("drafts/figs")
    rng = np.random.default_rng(1)
    facts = json.load(open("quant_runs/paper2_facts.json", encoding="utf-8"))
    fig, ax = plt.subplots(1, 2, figsize=(5.9, 2.19))

    # ---- (a) N = 20 上三种协议的 FP32 部署 MAE（8 seed 均值 ± sd）----
    keys = [("none", "live"), ("before_snap", "ema_before"), ("after_snap", "ema_after")]
    cols = ["#888888", "#7aa6c2", "#c98b8b"]
    ys = []
    for i, (m, _) in enumerate(keys):
        v = np.array([float(r["mae_fp32"]) for r in seq[(m, 20)]])
        ys.append(v)
        ax[0].scatter(np.full(v.size, i) + rng.uniform(-0.055, 0.055, v.size), v,
                      s=10, color=cols[i], alpha=0.9, lw=0, zorder=3)
        ax[0].plot([i - 0.15, i + 0.15], [v.mean()] * 2, color=cols[i], lw=1.5, zorder=4)
        ax[0].errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="none", ecolor=cols[i],
                       elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=4)
        ax[0].text(i, 0.0552, "%.4f" % v.mean(), ha="center", va="bottom",
                   fontsize=7.0, color="#222222")
    for a_, b_ in zip(ys[1], ys[2]):
        ax[0].plot([1, 2], [a_, b_], color="#9a9a9a", lw=0.5, alpha=0.7, zorder=2)
    d = float(np.mean(np.array(ys[1]) - np.array(ys[2])))
    ax[0].set_xticks(np.arange(3)); ax[0].set_xticklabels([lab for _, lab in keys])
    ax[0].set_xlabel("protocol at N = 20")
    ax[0].set_ylabel("FP32 task MAE")
    ax[0].set_ylim(0.0550, 0.0660)
    ax[0].set_title("(a) function space: ordering is flat\n(8 seeds, mean +- sd, paired)")
    ax[0].text(0.03, 0.995, "before - after = %+.4f (paired, n = 8)\npreset margin 0.0016" % d,
               transform=ax[0].transAxes, fontsize=6.9, color="#333333",
               va="top", ha="left", linespacing=1.35)
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
    p = out / "figS6.png"
    fig.savefig(p, dpi=660)
    plt.close(fig)
    return p


def main() -> None:
    rows = main_rows()
    seq = seq_rows()
    print("wrote", fig1_deploy_mae(rows, seq))
    print("wrote", fig2_weight_vs_function(seq))
    for k in SEQ:
        m4, s4 = stat(seq[k], "mae_4")
        f32, _ = stat(seq[k], "mae_fp32")
        print(f"  {k}: n={len(seq[k])} fp32={f32:.4f} mae4={m4:.4f} sd={s4:.4f}")


if __name__ == "__main__":
    main()