"""第二十二轮补充的三张配图（只读 quant_runs/，不改 markdown/docx/pdf）。

图 A  drafts/figs/figS4.png        —— 等价检验的功效曲线（df = 7，SE 取 1,556 张口径的实测值）。
图 B  drafts/figs/fig3.png  —— 随机舍入 8 个 seed 的恢复比例（held-out 1,556 张、8 bit）。
图 C  drafts/figs/figS3.png        —— 同一批 run 在 400 张与 1,556 张两套基数上的逐 seed 配对差。
数字全部来自 quant_runs/dual_basis_report.json、quant_runs/mechanism_stats.json、
quant_runs/heldout_eval_v3.csv，任何一步缺数直接抛错，不填占位值。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "drafts" / "figs"
FIG_W = 5.9
DPI = 830

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 7.4, "axes.titlesize": 7.9, "axes.labelsize": 7.4,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 7.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "lines.linewidth": 1.0, "lines.markersize": 3.4, "patch.linewidth": 0.6,
    "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

C_BLUE, C_ORANGE, C_GREEN, C_GREY = "#2f5d8a", "#a8541a", "#3f7a4a", "#7a7a7a"


def load_json(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def fig_power(dual):
    d_cell = dual["bases"]["heldout1556"]["d"]
    tcrit = stats.t.ppf(0.975, 7)
    se = (d_cell["ci"][1] - d_cell["ci"][0]) / (2 * tcrit)
    obs = d_cell["mean"]
    t_one = stats.t.ppf(0.95, 7)
    margins = np.linspace(0.00005, 0.0032, 400)
    # TOST：两个单侧检验各在 0.05 水平，两个拒绝域要同时成立（同一统计量，高度相关，不能相加）：
    # 拒绝 ⟺ |mean| < δ − t·SE；在真值 0 下 power = F(δ/SE − t) − F(−δ/SE + t)。
    power = np.array([stats.t.cdf(m / se - t_one, 7) - stats.t.cdf(-m / se + t_one, 7)
                      for m in margins])
    mde = float((tcrit + stats.t.ppf(0.80, 7)) * se)
    half = float(np.interp(0.5, power, margins))
    fig, ax = plt.subplots(figsize=(FIG_W * 0.78, 2.05))
    ax.plot(margins * 1000, power, color=C_BLUE, lw=1.2)
    ax.axvline(1.6, color=C_ORANGE, ls="--", lw=0.9)
    ax.annotate("preset margin 0.0016\n(power = 1.0)", xy=(1.6, 1.0), xytext=(1.78, 0.62),
                fontsize=7.0, color=C_ORANGE,
                arrowprops=dict(arrowstyle="-", color=C_ORANGE, lw=0.6))
    ax.axhline(0.8, color=C_GREY, ls=":", lw=0.8)
    ax.annotate("80 %% power, MDE = %.6f" % mde, xy=(mde * 1000, 0.8), xytext=(1.30, 0.83),
                fontsize=7.0, color=C_GREY)
    ax.annotate("half power at %.6f" % half, xy=(half * 1000, 0.5), xytext=(0.45, 0.30),
                fontsize=7.0, color=C_GREY,
                arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.6))
    ax.set_xlabel("equivalence margin $\\delta$ (MAE units × 1000)")
    ax.set_ylabel("power of TOST")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 1.04)
    ax.set_title("Equivalence test against a preset margin ($n$ = 8, df = 7)", pad=2.5)
    fig.savefig(OUT / "figS4.png")
    plt.close(fig)
    print("fig7 power: se=%.9f obs=%.9f half=%.6f mde=%.6f power@1.6=%.4f" %
          (se, obs, half, mde, float(np.interp(1.6, margins * 1000, power))))


def fig_sr(mech, held):
    cells = mech["table3"]["cells"]
    rn = np.array(cells["RN@1"]["b8"]["vals"])
    sr = np.array(cells["SR@1"]["b8"]["vals"])
    none = {}
    with (ROOT / held).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["mode"] == "none" and row["N"] == "0" and row["subset"] == "heldout_1556":
                none[int(row["seed"])] = float(row["mae_8"])
    assert sorted(none) == list(range(8)), sorted(none)
    base = np.array([none[s] for s in range(8)])
    rec = (rn - sr) / (rn - base)
    assert len(rn) == len(sr) == len(base) == 8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, 2.30),
                                   gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.40})
    x = np.arange(8)
    ax1.plot(x, rn, "o", color=C_ORANGE, label="nearest rounding, N = 1")
    ax1.plot(x, sr, "s", color=C_BLUE, label="stochastic rounding, N = 1")
    ax1.axhline(base.mean(), color=C_GREEN, ls="--", lw=0.9,
                label="projection off (mean of 8 seeds)")
    ax1.fill_between([-0.4, 7.4], base.mean() - base.std(ddof=1),
                     base.mean() + base.std(ddof=1), color=C_GREEN, alpha=0.12, lw=0)
    ax1.set_xticks(x)
    ax1.set_xlim(-0.4, 7.4)
    ax1.set_xlabel("initial seed")
    ax1.set_ylabel("deployment MAE (8 bit, 1,556 images)")
    ax1.set_ylim(0.0505, 0.0795)
    ax1.legend(loc="center left", frameon=False, handletextpad=0.3, borderpad=0.1)
    ax1.set_title("(a) rounding modes vs projection-off level", pad=2.5)
    ax2.axhline(1.0, color=C_GREY, ls=":", lw=0.8)
    ax2.bar(x, rec * 100, color=C_BLUE, width=0.68)
    ax2.axhline(rec.mean() * 100, color=C_ORANGE, ls="--", lw=0.9)
    ax2.annotate("mean %.1f %%" % (rec.mean() * 100), xy=(3.5, rec.mean() * 100),
                 xytext=(2.35, 108.5), fontsize=7.0, color=C_ORANGE)
    ax2.set_xticks(x)
    ax2.set_ylim(0, 116)
    ax2.set_xlabel("initial seed")
    ax2.set_ylabel("cost recovered (%)")
    ax2.set_title("(b) share of the cost removed", pad=2.5)
    fig.savefig(OUT / "fig3.png")
    plt.close(fig)
    print("fig8 sr: rec=%s mean=%.4f" % (np.round(rec, 3).tolist(), rec.mean()))


def fig_basis(dual):
    labels = {"a": "N = 1 vs off", "b": "N = 20 vs off",
              "c": "shadow vs live", "d": "before vs after"}
    colors = {"a": C_ORANGE, "b": C_BLUE, "c": C_GREEN, "d": C_GREY}
    fig, ax = plt.subplots(figsize=(FIG_W * 0.74, 2.55))
    allv = []
    for k in ["a", "b", "c", "d"]:
        f = np.array([dual["bases"]["first400"][k]["per_seed"][str(s)] for s in range(8)])
        h = np.array([dual["bases"]["heldout1556"][k]["per_seed"][str(s)] for s in range(8)])
        allv += f.tolist() + h.tolist()
        ax.plot(f * 1000, h * 1000, "o", ms=3.0, color=colors[k], label="%s: %s" % (k, labels[k]))
    lim = (min(allv) * 1000 - 1.2, max(allv) * 1000 + 1.2)
    ax.plot(lim, lim, color=C_GREY, ls="--", lw=0.7)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("paired difference on 400 images (× 1000)")
    ax.set_ylabel("paired difference on 1,556 images (× 1000)")
    ax.set_title("Two evaluation bases, one point per seed", pad=2.5)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.1)
    ax.axhline(0, color=C_GREY, lw=0.5)
    ax.axvline(0, color=C_GREY, lw=0.5)
    ax.annotate("above the diagonal: the 1,556-image\nbasis reports the larger difference",
                xy=(0, 0), xytext=(0.03, 0.06), textcoords="axes fraction", fontsize=6.9, color=C_GREY)
    fig.savefig(OUT / "figS3.png")
    plt.close(fig)
    print("fig9 basis: lim=%s" % (lim,))


def main():
    dual = load_json("quant_runs/dual_basis_report.json")
    mech = load_json("quant_runs/mechanism_stats.json")
    fig_power(dual)
    fig_sr(mech, "quant_runs/heldout_eval_v3.csv")
    fig_basis(dual)


if __name__ == "__main__":
    main()
