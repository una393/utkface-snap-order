"""篇 2 的两张补充配图（只读数据，不改任何 markdown/docx/pdf）。

图 A  drafts/figs/figS5.png     —— 4 bit 崩塌由 run 抽签主导。
      数据源 quant_runs/deploy_table_v2.csv；画前把 21 个同配置 run 逐格核对验收基线
      TABLE_5_4，任何一格不符直接抛错，不改表也不改图去凑。
图 B  drafts/figs/figS7.png —— 从导出权重到板端指令流的编译链路与两类误差量级。
      数字全部来自 npu_quant/REPORT.md，出处行号写在各常量旁。图内文字全英文。
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.text import Text

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "drafts" / "figs"
FIG_W = 5.9          # in，按 14.5 cm 插入时缩放 0.9676
DPI = 660
PRINT_CM = 14.5

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 7.4, "axes.titlesize": 7.9, "axes.labelsize": 7.4,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 7.4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "lines.linewidth": 1.0, "lines.markersize": 3.4, "patch.linewidth": 0.6,
    "text.color": "#111111", "axes.edgecolor": "#3a3a3a",
    "axes.labelcolor": "#111111", "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

# 验收基线：间隔 | n | 均值 | 范围 | sd（21 个同配置 run，跨 none/before_snap/after_snap）
TABLE_5_4 = {1: (5, 0.1224, 0.0899, 0.2337, 0.0626),
             5: (9, 0.1248, 0.0857, 0.2117, 0.0510),
             20: (7, 0.1315, 0.0849, 0.2030, 0.0476)}

SEED = {"0": ("#1f4e79", "o"), "1": ("#b34a00", "s"), "2": ("#2d7a3e", "^")}


# ---------------------------------------------------------------- 验收工具
def audit(fig: Figure) -> dict:
    """渲染前量一遍：最小字号、等效印刷字号、文字框重叠、越界裁切、中文。"""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tlist = [t for t in fig.findobj(match=Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t.get_text().replace("\n", " | "), t.get_window_extent(renderer=rend)) for t in tlist]
    w_in, h_in = fig.get_size_inches()
    min_pt = min(t.get_fontsize() for t in tlist)
    eq_pt = min_pt * (PRINT_CM / 2.54) / w_in

    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 0.5 and oy > 0.5:
                overlaps.append((boxes[i][0], boxes[j][0], round(ox, 1), round(oy, 1)))

    fb = fig.bbox
    outside = [(txt, round(b.x0 - fb.x0, 1), round(fb.x1 - b.x1, 1), round(b.y0 - fb.y0, 1),
                round(fb.y1 - b.y1, 1))
               for txt, b in boxes
               if b.x0 < fb.x0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y0 < fb.y0 - 0.5 or b.y1 > fb.y1 + 0.5]

    cjk = [txt for txt, _ in boxes if re.search(r"[\u3000-\u9fff\uff00-\uffef]", txt)]
    return {"texts": len(tlist), "min_pt": min_pt, "eq_pt": eq_pt, "overlaps": overlaps,
            "outside": outside, "cjk": cjk}


_ORIG_SAVEFIG = Figure.savefig
AUDIT_LOG: list[tuple[str, dict]] = []


def _savefig_audited(self, fname, *a, **kw):  # noqa: ANN001
    AUDIT_LOG.append((str(fname), audit(self)))
    return _ORIG_SAVEFIG(self, fname, *a, **kw)


Figure.savefig = _savefig_audited


# ---------------------------------------------------------------- 图 A
def fig_a(rows: list[dict]) -> None:
    """4 bit 散点：数据源 quant_runs/deploy_table_v2.csv。

    只取同配置 seed 重复（ema_decay == 0.99 且 split_seed == seed）；γ 扫描 4 个 run、
    固定划分 3 个 run、snap_every=0 的两个 N=∞ run 都属别的因子，全部排除。
    应得 N=1 共 5 个、N=5 共 9 个、N=20 共 7 个，计数不符直接抛错，不改图去凑。
    """
    sel = [r for r in rows
           if int(r["snap_every"]) != 0
           and float(r["ema_decay"]) == 0.99 and r["split_seed"].strip() == r["seed"].strip()]
    grp = {f: sorted((r for r in sel if int(r["snap_every"]) == f), key=lambda r: (r["mode"], r["seed"]))
           for f in (1, 5, 20)}
    stat = {f: (np.array([float(r["mae_4"]) for r in grp[f]]),
                np.array([float(r["mae_fp32"]) for r in grp[f]])) for f in (1, 5, 20)}

    print("=== 图 A 筛选：ema_decay == 0.99 且 split_seed == seed ===")
    for f in (1, 5, 20):
        v, fp = stat[f]
        print(f"  N={f:>2}  n={len(v)}  mean={v.mean():.4f}  sd={v.std(ddof=1):.4f}"
              f"  range={v.min():.4f}-{v.max():.4f}  FP32 group mean={fp.mean():.4f}")
    print(f"  total runs = {len(sel)}")
    assert {f: len(grp[f]) for f in (1, 5, 20)} == {1: 5, 5: 9, 20: 7}, \
        f"筛选后 n 不是 5/9/7：{ {f: len(grp[f]) for f in (1, 5, 20)} }"

    mus = [float(stat[f][0].mean()) for f in (1, 5, 20)]
    sds = [float(stat[f][0].std(ddof=1)) for f in (1, 5, 20)]
    assert [round(m, 4) for m in mus] == [0.1224, 0.1248, 0.1315], f"组均值与验收不符：{mus}"
    assert [round(s, 4) for s in sds] == [0.0626, 0.0510, 0.0476], f"组 sd 与验收不符：{sds}"

    pos = {1: 0.0, 5: 1.0, 20: 2.0}
    fig, ax = plt.subplots(figsize=(FIG_W, 3.15))
    fig.subplots_adjust(left=0.118, right=0.985, bottom=0.205, top=0.895)

    for f in (1, 5, 20):
        x = pos[f]
        rs = grp[f]
        v, fp = stat[f]
        mu, sd = float(v.mean()), float(v.std(ddof=1))

        ax.plot([x - 0.34, x + 0.34], [fp.mean(), fp.mean()], color="#8a8a8a", lw=0.9,
                ls=(0, (3.0, 2.0)), zorder=2)
        ax.errorbar([x], [mu], yerr=[[sd], [sd]], fmt="none", ecolor="#111111",
                    elinewidth=0.9, capsize=3.0, capthick=0.9, zorder=3)
        ax.plot([x - 0.34, x + 0.34], [mu, mu], color="#111111", lw=1.6, solid_capstyle="butt", zorder=5)
        ax.text(x + 0.37, mu, f"{mu:.4f}", ha="left", va="center", fontsize=7.4)
        ax.text(x, float(v.max()) + 0.0060, f"n = {len(rs)}", ha="center", va="bottom",
                fontsize=7.4, color="#555555")
        for off, r in zip(np.linspace(-0.215, 0.215, len(rs)), rs):
            c, m = SEED[r["seed"]]
            ax.scatter([x + off], [float(r["mae_4"])], s=21, marker=m, facecolor=c,
                       edgecolor="white", linewidth=0.4, zorder=4)

    ax.set_xlim(-0.60, 2.62)
    ax.set_ylim(0.038, 0.285)
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_xticklabels(["1", "5", "20"])
    ax.set_yticks(np.arange(0.050, 0.251, 0.025))
    ax.set_xlabel("snap interval N (steps)")
    ax.set_ylabel("task MAE after 4-bit\nquantization")
    ax.set_title("At 4 bits, run-to-run spread dominates the interval")
    ax.grid(True, axis="y", alpha=0.22, lw=0.4)

    handles = [Line2D([], [], marker=SEED[s][1], ls="none", color=SEED[s][0], markersize=3.6,
                      markeredgecolor="white", markeredgewidth=0.4, label=f"seed {s}") for s in ("0", "1", "2")]
    handles.append(Line2D([], [], color="#111111", lw=1.6, label="mean \u00b1 1 sd"))
    handles.append(Line2D([], [], color="#8a8a8a", lw=0.9, ls=(0, (3.0, 2.0)),
                          label="group mean of FP32"))
    ax.legend(handles=handles, loc="upper right", ncol=2, frameon=True, framealpha=0.92,
              borderpad=0.3, handletextpad=0.4, columnspacing=0.9, labelspacing=0.3,
              edgecolor="#d0d0d0")

    fig.text(0.5, 0.073, f"Between-interval means span {min(mus):.4f} \u2013 {max(mus):.4f};",
             ha="center", va="center", fontsize=7.4)
    fig.text(0.5, 0.026, f"within-interval sd spans {min(sds):.4f} \u2013 {max(sds):.4f}.",
             ha="center", va="center", fontsize=7.4)

    fig.savefig(OUT / "figS5.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- 图 B
def fig_b() -> None:
    """编译链路 + 两类误差量级（柱）+ b30.lut 一元算子（chips）。图内全是英文。"""
    fig = plt.figure(figsize=(FIG_W, 3.45))

    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # REPORT.md 行 277–280（after_snap ema）、行 159（1x1x3x224x224）、行 27/256（100 张校准）、
    # 行 102/217（激活 per-tensor percentile）、行 112（.hbm 6.47–6.49 MB）、行 9/160（nash-m）、
    # 行 169–170/30（200 张测试）、行 164–165/246（x86 指令级仿真器）
    stages = [
        ("QAT ckpt", "best.pth\nafter_snap ema"),
        ("ONNX export", "static graph\n1x3x224x224"),
        ("hmct PTQ", "per-tensor\n100 calib imgs"),
        ("hbdk4", "compile .hbm\n6.47-6.49 MB"),
        ("hbrt4 nash", "simulator\n200 test imgs"),
    ]
    x0, step, bw, bh, by = 0.018, 0.196, 0.176, 0.168, 0.726
    ax.add_patch(FancyBboxPatch((0.008, 0.706), 2 * step - 0.010, 0.252,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor="#e8eff7", edgecolor="none", zorder=0))
    ax.add_patch(FancyBboxPatch((x0 + 2 * step - 0.012, 0.706), 0.600, 0.252,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor="#fbeee2", edgecolor="none", zorder=0))
    for i, (name, det) in enumerate(stages):
        bx = x0 + i * step
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.010",
                                    facecolor="white", edgecolor="#3a3a3a", lw=0.7, zorder=2))
        ax.text(bx + bw / 2, by + bh * 0.70, name, ha="center", va="center",
                fontsize=7.6, zorder=3)
        ax.text(bx + bw / 2, by + bh * 0.28, det, ha="center", va="center",
                fontsize=7.4, linespacing=1.25, zorder=3)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((bx + bw + 0.003, by + bh / 2), (bx + step - 0.004, by + bh / 2),
                                         arrowstyle="-|>", mutation_scale=5, lw=0.7,
                                         color="#3a3a3a", shrinkA=0, shrinkB=0, zorder=3))
    ax.text(0.020, 0.930, "weights only (old simulation)", ha="left", va="center", fontsize=7.4)
    ax.text(x0 + 2 * step + 0.002, 0.930, "compiled int8 pipeline", ha="left", va="center",
            fontsize=7.4, fontweight="bold")
    ax.text(0.5, 0.985, "Where int8 error enters the deployment chain", ha="center", va="center",
            fontsize=7.9)

    # 柱：仅权重量化 ≤0.0003（REPORT 18–20、212）；ReLU-QAT N=1 +0.0038 [+0.0000, +0.0076]、
    # N=20 +0.0034 [+0.0005, +0.0062]、GLU +0.0042 [−0.0001, +0.0086]、GELU −0.0002 [−0.0038, +0.0034]；
    # 与稿 §5.7 表 8 同口径（同 200 张逐图配对），误差棒是 95 % CI 而非标准误
    bar = fig.add_axes([0.185, 0.455, 0.795, 0.225])
    cats = ["weights only", "ReLU-QAT N=1", "ReLU-QAT N=20", "ReLU-QAT N=inf", "GLU N=1", "GELU N=1"]
    vals = [0.0003, 0.003834, 0.003368, 0.004025, 0.004233, -0.000157]
    lo = [0.0003, 0.00004, 0.00054, -0.001201, -0.00011, -0.00376]
    hi = [0.0003, 0.00763, 0.00620, 0.009252, 0.00858, 0.00344]
    cols = ["#8fa9c4", "#c06a1c", "#c06a1c", "#a8541a", "#c06a1c", "#c06a1c"]
    ypos = np.arange(len(cats))[::-1]
    bar.barh(ypos, vals, height=0.50, color=cols, edgecolor="#3a3a3a", lw=0.6, zorder=3)
    xerr = np.array([[v - a for v, a in zip(vals, lo)], [b - v for v, b in zip(vals, hi)]])
    bar.errorbar(vals, ypos, xerr=xerr, fmt="none", ecolor="#222222", elinewidth=0.7,
                 capsize=1.8, capthick=0.7, zorder=4)
    bar.axvline(0.0, color="#3a3a3a", lw=0.8, ls=(0, (3, 2)), zorder=2)
    bar.set_yticks(ypos)
    bar.set_yticklabels(cats, fontsize=7.4)
    bar.set_xlim(-0.0050, 0.0108)
    bar.set_ylim(-0.62, 5.62)
    bar.set_xticks([-0.004, -0.002, 0.0, 0.002, 0.004, 0.006, 0.008])
    bar.set_xticklabels(["−0.004", "−0.002", "0", "0.002", "0.004", "0.006", "0.008"], fontsize=7.2)
    bar.tick_params(axis="x", pad=1.5, length=2.2)
    bar.tick_params(axis="y", pad=1.8, length=0)
    bar.set_xlabel("paired \u0394MAE vs FP32 on 200 test images (bars: 95 % CI)", fontsize=7.2, labelpad=2.0)
    for sp in ("top", "right"):
        bar.spines[sp].set_visible(False)
    bar.grid(axis="x", color="#dcdcdc", lw=0.5, zorder=0)
    for v, a, b, y in zip(vals, lo, hi, ypos):
        if a == b:
            bar.text(v + 0.00018, y, "\u2264 0.0003", ha="left", va="center", fontsize=7.4)
        else:
            bar.text(b + 0.00018, y, ("%+.4f" % v).replace("-", "\u2212"),
                     ha="left", va="center", fontsize=7.4)

    # REPORT 行 134 / 299–301：控制头内的 b30.lut 指令数
    ax.text(0.018, 0.342, "unary ops lowered to b30.lut, per control head", ha="left", va="center",
            fontsize=7.4)
    chips = [("ReLU-QAT", "3", ["Pow", "HzRsqrt", "ReLU"]),
             ("GELU", "2", ["Tanh", "HzGelu"]),
             ("Bilinear-GLU", "0", [])]
    cw, ch, gap = 0.155, 0.060, 0.012
    for k, (head, n, ops) in enumerate(chips):
        yy = 0.262 - k * 0.092
        ax.text(0.018, yy, head, ha="left", va="center", fontsize=7.4)
        ax.text(0.151, yy, n, ha="right", va="center", fontsize=7.6, fontweight="bold")
        if ops:
            for j, op in enumerate(ops):
                cx = 0.176 + j * (cw + gap)
                ax.add_patch(FancyBboxPatch((cx, yy - ch / 2), cw, ch,
                                            boxstyle="round,pad=0,rounding_size=0.012",
                                            facecolor="#f2f2f2", edgecolor="#8a8a8a", lw=0.6, zorder=2))
                ax.text(cx + cw / 2, yy, op, ha="center", va="center", fontsize=7.4, zorder=3)
        else:
            ax.add_patch(FancyBboxPatch((0.176, yy - ch / 2), cw, ch,
                                        boxstyle="round,pad=0,rounding_size=0.012",
                                        facecolor="white", edgecolor="#b0b0b0",
                                        lw=0.6, ls=(0, (2, 2)), zorder=2))
            ax.text(0.176 + cw / 2, yy, "live", ha="center", va="center", fontsize=7.4,
                    color="#6a6a6a", zorder=3)

    fig.savefig(OUT / "figS7.png", dpi=DPI)
    plt.close(fig)


def read_table() -> list[dict]:
    """读 deploy_table_v2.csv 的 30 行；不绘图，便于单独验证入口。"""
    rows = list(csv.DictReader(open(ROOT / "quant_runs" / "deploy_table_v2.csv", encoding="utf-8")))
    assert len(rows) == 30, f"deploy_table_v2.csv 行数 {len(rows)} != 30"
    return rows


def check_table(rows: list[dict]) -> None:
    """21 个同配置 run 复算 vs 验收基线 TABLE_5_4，任何一格不符直接抛错。"""
    sel = [r for r in rows if int(r["snap_every"]) != 0
           and float(r["ema_decay"]) == 0.99 and r["split_seed"].strip() == r["seed"].strip()]
    print("=== 图 A：21 个同配置 run 复算 vs 验收基线 TABLE_5_4 ===")
    bad = []
    for f, (n_t, mu_t, lo_t, hi_t, sd_t) in TABLE_5_4.items():
        v = np.array([float(r["mae_4"]) for r in sel if int(r["snap_every"]) == f])
        n, mu, lo, hi, sd = len(v), v.mean(), v.min(), v.max(), v.std(ddof=1)
        fp = np.array([float(r["mae_fp32"]) for r in sel if int(r["snap_every"]) == f])
        got = (n, round(mu, 4), round(lo, 4), round(hi, 4), round(sd, 4))
        exp = (n_t, mu_t, lo_t, hi_t, sd_t)
        ok = got == exp
        print(f"  N={f:>2}  n={n}  mean={mu:.5f}  range={lo:.4f}-{hi:.4f}  sd={sd:.5f}"
              f"  | 基线={exp} 复算={got}  {'OK' if ok else 'DIFF'}"
              f"  | FP32 ref mean={fp.mean():.5f}")
        if not ok:
            bad.append((f, exp, got))
    if bad:
        raise SystemExit(f"CSV 与验收基线不符，停止出图：{bad}")


def main() -> None:
    rows = read_table()
    check_table(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    fig_a(rows)
    fig_b()

    print("\n=== 出图验收 ===")
    for path, info in AUDIT_LOG:
        px = plt.imread(path).shape
        fig_w = None
        print(f"\n  {path}")
        print(f"    pixels = {px[1]} x {px[0]}   (dpi={DPI})")
        print(f"    最小字号 = {info['min_pt']:.2f} pt → 等效印刷 "
              f"{info['eq_pt']:.2f} pt @ {PRINT_CM} cm 宽")
        print(f"    Text 对象 = {info['texts']}；重叠对 = {len(info['overlaps'])}；"
              f"越界 = {len(info['outside'])}；含中文 = {len(info['cjk'])}")
        for o in info["overlaps"]:
            print(f"      OVERLAP {o[0]!r} x {o[1]!r}  ({o[2]}x{o[3]} px)")
        for o in info["outside"]:
            print(f"      OUTSIDE {o[0]!r}  margins px L={o[1]} R={o[2]} B={o[3]} T={o[4]}")
        for o in info["cjk"]:
            print(f"      CJK {o!r}")


if __name__ == "__main__":
    main()
