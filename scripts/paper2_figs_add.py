"""篇 2 补充的三张图：协议示意、收敛曲线、部署网格残差分布。

run 目录一律由 quant_runs/deploy_table_v2.csv 的 run 列解析（见 run_index()），
不按目录名猜：results/ 里有同 (head, mode, seed, snap_every) 但不同 ema_decay 的
重复 run，目录名分不出谁是谁。history.csv 出收敛曲线，best.pth 出的权重算网格残差。

权重张量与残差口径直接复用 scripts/paper2_diagnostics.py（quant_tensors /
grid_stats / load_run），所以图 5 的数与 quant_runs/paper2_facts.json、草稿 §5.1
的表同源，不另立一套口径。
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
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import MaxNLocator

from scripts.paper2_diagnostics import grid_stats, load_run, quant_tensors

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.unicode_minus"] = False
# 按印刷尺寸出图：宽 5.7–5.9 in（14.5 cm 下缩放 0.97–1.00），字号 7.4–8.6 pt
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.6, "axes.labelsize": 7.5,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 7.4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "lines.linewidth": 1.1, "patch.linewidth": 0.6,
})

HEAD = "relu_qat"
SEED = 0
FREQS = [1, 5, 20]
COL = {1: "#2f6fa8", 5: "#c8811f", 20: "#a8322f"}
DPI = 660
OUT = Path("drafts/figs")

# 草稿 §5.1 表的区间（relu_qat / seed 0 / after_snap 口径），只做一致性检查，
# 不参与画图、不改任何数。
DRAFT_51 = {1: ((0.0, 0.0000), (1.000, 1.000)),
            5: ((0.0000, 0.0004), (0.951, 1.000)),
            20: ((0.0034, 0.0037), (0.571, 0.618))}


DEPLOY_TABLE = Path("quant_runs/deploy_table_v2.csv")


def deploy_rows() -> list[dict]:
    """deploy_table_v2.csv 的全部行；run 选择的唯一权威来源。"""
    if not DEPLOY_TABLE.exists():
        raise SystemExit(f"missing {DEPLOY_TABLE}: run selection needs the deploy table, no guessing")
    return list(csv.DictReader(DEPLOY_TABLE.open(encoding="utf-8")))


def is_main_config(r: dict) -> bool:
    """主配置 = ema_decay 0.99 且 split_seed == seed；其余行是补的参数扫描（0.9 / 0.999）。"""
    return float(r["ema_decay"]) == 0.99 and int(r["split_seed"]) == int(r["seed"])


def run_index(allow_missing: frozenset[str] = frozenset()) -> dict[tuple[str, str, int, int], Path]:
    """(head, ema_mode, seed, snap_every) -> run 目录，只收主配置行。

    run 目录名由 deploy_table_v2.csv 的 run 列定位，表里找不到就抛错，不回退到目录名
    猜：results/ 下有同 (head, mode, seed, snap_every) 但不同 ema_decay 的重复 run，
    目录名本身分不出谁是谁。表里没有的 head（gelu / bilinear_glu）因此不在本索引内。
    """
    idx: dict[tuple[str, str, int, int], Path] = {}
    for r in deploy_rows():
        if not is_main_config(r):
            continue
        d = Path("results") / r["run"]
        rj = d / "result.json"
        if not rj.exists():
            if d.name in allow_missing:
                print(f"[skip] {d.name}: run directory was removed from the workspace")
                continue
            raise SystemExit(f"deploy_table_v2.csv names {d.name}, but {rj} is missing")
        a = json.loads(rj.read_text(encoding="utf-8"))["args"]
        fp = d / "fingerprint.json"
        cfg = json.loads(fp.read_text(encoding="utf-8")).get("config", {}) if fp.exists() else {}
        if "snap_every" in a and "snap_every" in cfg and a["snap_every"] != cfg["snap_every"]:
            raise SystemExit(f"snap_every disagrees between result.json and fingerprint: {d.name}")
        if (a["ema_mode"], int(a.get("seed", 0))) != (r["mode"], int(r["seed"])):
            raise SystemExit(f"table row disagrees with {d.name}: {r['mode']}/s{r['seed']}")
        n_run = a.get("snap_every", cfg.get("snap_every"))
        if n_run is not None and int(n_run) != int(r["snap_every"]):
            raise SystemExit(f"snap_every disagrees for {d.name}: table {r['snap_every']} vs run {n_run}")
        key = (a["head"], a["ema_mode"], int(r["seed"]), int(r["snap_every"]))
        if key in idx:
            raise SystemExit(f"ambiguous main-config run for {key}: {idx[key].name} vs {d.name}")
        idx[key] = d
    return idx


def pick(idx, head, mode, seed, n) -> tuple[Path, str]:
    """找不到 (head, mode, seed, n) 时退回 after_snap，并返回实际用的协议名。"""
    if (head, mode, seed, n) in idx:
        return idx[(head, mode, seed, n)], mode
    if (head, "after_snap", seed, n) in idx:
        return idx[(head, "after_snap", seed, n)], "after_snap"
    raise SystemExit(f"no run for {head}/{mode}/s{seed}/N={n} and no after_snap fallback")


def meta_of(d: Path) -> dict:
    return json.loads((d / "result.json").read_text(encoding="utf-8"))


def read_history(d: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(d / "history.csv")))
    ep = np.array([int(r["epoch"]) for r in rows])
    live = np.array([float(r["val_mae_live"]) for r in rows])
    order = np.argsort(ep)
    return ep[order], live[order]


def resid_grid(w, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """逐元素网格残差 rho = |q*scale - w| / scale（单位：格距），与 grid_stats 同口径。

    同时返回以权重为单位的绝对残差 delta = |q*scale - w|，供调用方复核 RMS 口径。
    """
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    q = (flat / scale).round().clamp(-qmax - 1, qmax)
    delta = (q * scale - flat).abs()
    return (delta / scale).flatten().numpy(), delta.flatten().numpy()


# ---------------------------------------------------------------- 图 3：协议示意
def _box(ax, cx, cy, w, h, text, fc, fs=7.5, ec="#3b3b3b", ls="solid"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.07",
                                fc=fc, ec=ec, lw=1.0, ls=ls, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.4)


def _arrow(ax, x0, y0, x1, y1, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#3b3b3b", shrinkA=0, shrinkB=0,
                                mutation_scale=7, connectionstyle=f"arc3,rad={rad}"), zorder=1)


def fig3_protocols() -> Path:
    # 5.9 x 3.72 in（四行：三种覆写协议 + STE 对照）；
    # 1 数据单位 = 0.590 in，两轴同比；行距 1.48 使影子行（+0.80）与其上一行留出 0.18 的空白，
    # 否则 before_snap 的 E 与影子导出框会贴住 none 行、读成本行的一部分。
    fig, ax = plt.subplots(figsize=(5.9, 3.72))
    ax.set_xlim(0, 10.0); ax.set_ylim(0.0, 6.30); ax.axis("off")

    rows = {"none": 5.10, "before_snap": 3.62, "after_snap": 2.14, "ste": 0.66}
    LAND, LH, TH = 0.80, 0.50, 0.50
    x_th, w_th = 2.00, 1.42
    x_s, w_s = 3.915, 2.05
    x_e_b, x_e_a, w_e = 3.10, 6.05, 3.40
    x_exp, w_exp = 9.09, 1.62

    for i, (mode, y) in enumerate(rows.items()):
        if i % 2 == 0:   # 隔行浅底：把该行的 θ_t、S 与其上方的 E/导出框绑成一组
            ax.add_patch(Rectangle((0.0, y - 0.34), 10.0, 1.46, fc="#f3f5f8", ec="none", zorder=0))
        ax.text(0.06, y, mode, ha="left", va="center", fontsize=7.6, fontweight="bold")
        _box(ax, x_th, y, w_th, TH, "$\\theta_t$\nlive weights", "#eef1f4")
        s_txt = ("$S$ in forward only\nint8, per out-channel" if mode == "ste"
                 else "$S$: snap_weights_()\nint8, per out-channel")
        _box(ax, x_s, y, w_s, TH, s_txt, "#dce7f1")
        _arrow(ax, x_th + w_th / 2, y, x_s - w_s / 2, y)

        if mode == "none":
            _box(ax, 6.55, y + LAND, 2.20, LH,
                 "no shadow weight\n$E$ is never applied", "#f2f2f2", ec="#9a9a9a", ls=(0, (4, 3)))
            _box(ax, x_exp, y, w_exp, TH, "exported weight\n$= \\theta_t$ (live)", "#f6ecec")
            _arrow(ax, x_s + w_s / 2, y, x_exp - w_exp / 2, y)
        elif mode == "before_snap":
            _box(ax, x_e_b, y + LAND, w_e, LH,
                 "$E$: $\\bar\\theta \\leftarrow \\gamma\\bar\\theta + (1-\\gamma)\\theta_t$\n"
                 "on the pre-snap weights (before $S$)", "#f3e6cf")
            _arrow(ax, x_th, y + TH / 2, x_th, y + LAND - LH / 2)
            _box(ax, x_exp, y + LAND, w_exp, LH, "exported weight\n$= \\bar\\theta$ (shadow)", "#f6ecec")
            _arrow(ax, x_e_b + w_e / 2, y + LAND, x_exp - w_exp / 2, y + LAND)
            ax.text(5.10, y, "live weights for step $t+1$", ha="left", va="center",
                    fontsize=7.4, color="#6a6a6a")
        elif mode == "ste":
            _box(ax, x_exp, y, w_exp, TH, "exported weight\n$= \\theta_t$ (live)", "#f6ecec")
            _arrow(ax, x_s + w_s / 2, y, x_exp - w_exp / 2, y)
            ax.text(5.10, y + 0.26, "no write-back: $w$ is never assigned",
                    ha="left", va="bottom", fontsize=7.4, color="#6a6a6a")
        else:
            _box(ax, x_e_a, y + LAND, w_e, LH,
                 "$E$: $\\bar\\theta \\leftarrow \\gamma\\bar\\theta + (1-\\gamma)\\hat\\theta_t$\n"
                 "on the post-snap weights (after $S$)", "#f3e6cf")
            _arrow(ax, 4.55, y + TH / 2, 4.55, y + LAND - LH / 2)
            _box(ax, x_exp, y + LAND, w_exp, LH, "exported weight\n$= \\bar\\theta$ (shadow)", "#f6ecec")
            _arrow(ax, x_e_a + w_e / 2, y + LAND, x_exp - w_exp / 2, y + LAND)
            ax.text(5.10, y, "live weights for step $t+1$", ha="left", va="center",
                    fontsize=7.4, color="#6a6a6a")

    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.005)
    p = OUT / "fig1.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


# ------------------------------------------------------------ 图 4：收敛（val MAE）
def fig4_convergence(idx=None) -> tuple[Path, list[dict], float]:
    # idx 保留以兼容 main() 的调用；run 目录改由 deploy_table_v2.csv 的 run 列解析（见下）。
    # 5.7 x 3.43 in：正文按 12.0 cm 插图时缩放 0.829，字底 8.6 pt -> 7.1 pt 印刷
    fig, ax = plt.subplots(figsize=(5.7, 3.43))
    # 本图只用 mode=none 的行（每 N 唯一），但仍按主配置过滤：同一 N 也有 ema_decay
    # 补跑，少这层过滤就成了「CSV 里最后一行赢」，选谁取决于行序。
    runs: dict[int, Path] = {}
    for r in deploy_rows():
        if r["mode"] != "none" or int(r["seed"]) != SEED or not is_main_config(r):
            continue
        d = Path("results") / r["run"]
        if not (d / "history.csv").exists():
            raise SystemExit(f"no history.csv in {d}")
        runs[int(r["snap_every"])] = d
    if sorted(runs) != [0] + FREQS:
        raise SystemExit(f"deploy_table_v2.csv: unexpected N coverage {sorted(runs)}")

    C_INF = "#8b5a9e"          # 第 4 条曲线（N = inf）的颜色，与 COL 的三色可区分
    used, baselines = [], []
    for n in FREQS:
        d, proto = runs[n], "none"
        ep, live = read_history(d)
        m = meta_of(d)
        baselines.append(m["zero_baseline_mae"])
        i = int(np.argmin(live))
        ax.plot(ep, live, "-o", ms=3.6, lw=1.2, color=COL[n],
                label=f"$N={n}$ (protocol: {proto}, best {live[i]:.4f})")
        ax.plot(ep[i], live[i], "o", ms=6.0, color=COL[n], mec="white", mew=0.7, zorder=4)
        used.append({"N": n, "run": str(d), "protocol": proto, "best_epoch": int(ep[i]),
                     "best_mae": float(live[i]), "final_mae": float(live[-1]),
                     "zero_baseline": m["zero_baseline_mae"]})

    # N = inf：训练期完全不投影（none 协议），单独一条曲线
    d = runs[0]
    ep_inf, live_inf = read_history(d)
    m_inf = meta_of(d)
    baselines.append(m_inf["zero_baseline_mae"])
    i_inf = int(np.argmin(live_inf))
    ax.plot(ep_inf, live_inf, "-o", ms=3.6, lw=1.2, color=C_INF, label="N = inf (no snap)")
    ax.plot(ep_inf[i_inf], live_inf[i_inf], "o", ms=6.0, color=C_INF, mec="white", mew=0.7, zorder=4)
    used.append({"N": "inf", "run": str(d), "protocol": "none", "best_epoch": int(ep_inf[i_inf]),
                 "best_mae": float(live_inf[i_inf]), "final_mae": float(live_inf[-1]),
                 "zero_baseline": m_inf["zero_baseline_mae"]})

    if max(baselines) - min(baselines) > 1e-9:
        raise SystemExit(f"zero baseline differs across runs: {baselines}")
    zb = baselines[0]
    ax.axhline(zb, ls="--", lw=1.1, color="#555555",
               label=f"zero baseline, predict the mean = {zb:.4f}")

    ax.set_xlabel("epoch", fontsize=8.6)
    ax.set_ylabel("validation MAE (live weights)", fontsize=8.6)
    ax.set_xlim(-0.4, float(ep.max()) + 0.9)   # 右端留白，末个 epoch 刻度不出框
    ax.set_ylim(top=zb + 0.004)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=8.6)
    ax.legend(fontsize=8.6, loc="center right", framealpha=0.95)
    fig.tight_layout(pad=0.3)
    p = OUT / "figS1.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p, used, zb


# ------------------------------------------------------- 图 5：部署网格残差分布
LO, HI = 1e-8, 0.501          # rho 的显示区间；rho=0 落在最左一格


def fig5_grid_residual(idx) -> tuple[Path, list[dict]]:
    # 5.9 x 3.76 in：14.5 cm 插图时缩放 0.97，字底 7.4 pt -> 7.2 pt 印刷
    fig, ax = plt.subplots(figsize=(5.9, 3.76))
    bins = np.logspace(np.log10(LO), np.log10(HI), 71)
    groups = []
    for n in FREQS:
        d, proto = pick(idx, HEAD, "after_snap", SEED, n)
        key = "ema_state_dict" if proto != "none" else "model_state_dict"
        w = load_run(d, key)
        num = sum(v.numel() for v in w.values())
        # 与 paper2_diagnostics.main() 完全相同的合成方式：按元素数加权
        rms = sum(grid_stats(v)[0] * v.numel() / num for v in w.values())
        on_grid = sum(grid_stats(v)[1] * v.numel() / num for v in w.values())

        rho_all = []
        for name, v in w.items():
            rho, delta = resid_grid(v)
            rms_v, on_v = grid_stats(v)
            rms_from_rho = float(np.sqrt((delta.astype(np.float64) ** 2).sum()
                                         / (v.float() ** 2).sum()))
            if not (np.isclose(rms_from_rho, rms_v, rtol=1e-5, atol=1e-12)
                    and np.isclose(float((rho < 0.05).mean()), on_v, atol=1e-9)):
                raise SystemExit(f"residual scope disagrees with grid_stats on {name}")
            rho_all.append(rho)
        rho_all = np.concatenate(rho_all)
        if rho_all.max() > HI:
            raise SystemExit(f"N={n}: residual {rho_all.max()} above the plotted range")

        ax.hist(np.clip(rho_all, LO, HI), bins=bins, histtype="step", lw=1.2, color=COL[n],
                label=f"$N={n}$: rel. RMS {rms:.4f}, on-grid {(rho_all < 0.05).mean():.3f}, "
                      f"$\\rho=0$: {(rho_all == 0).mean():.1%}")
        groups.append({"N": n, "run": str(d), "protocol": proto,
                       "rms": float(rms), "on_grid": float(on_grid),
                       "n_tensors": len(w), "n_elements": int(num),
                       "on_grid_from_rho": float((rho_all < 0.05).mean()),
                       "frac_exact_zero": float((rho_all == 0).mean()),
                       "frac_gt_045": float((rho_all > 0.45).mean()),
                       "median_nonzero": float(np.median(rho_all[rho_all > 0])),
                       "max": float(rho_all.max())})

    ax.set_xscale("log")
    ax.axvline(0.05, color="#333333", ls="--", lw=0.9)
    ax.text(0.052, 0.055, r"$\tau=0.05$: on-grid criterion", rotation=90,
            transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=7.4)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 1.4e6)   # 顶端留给两行说明文字
    ax.set_xlim(LO, HI)
    ax.set_xlabel(r"grid residual $\rho=|w-S_8(w)|/\Delta_s$"
                  r"  (grid steps, log scale; $\rho=0$ falls at the left edge)")
    ax.set_ylabel("number of weights (log scale)")
    ax.grid(alpha=0.22, axis="y", which="major")
    ax.legend(fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=1,
              frameon=False, handlelength=2.6, borderaxespad=0.0,
              labelspacing=0.35)
    ax.text(0.012, 0.985, f"{groups[0]['n_tensors']} weight tensors, "
                          f"{groups[0]['n_elements']:,} weights per group;\n"
                          r"the leftmost bin also holds $0<\rho<10^{-8}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.4, color="#5a5a5a")
    fig.subplots_adjust(left=0.115, right=0.985, top=0.977, bottom=0.245)
    p = OUT / "figS2.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p, groups


def report_51(groups: list[dict]) -> bool:
    print("\n[check] vs draft section 5.1 (relu_qat / seed 0 / after_snap)")
    ok = True
    for g in groups:
        (rlo, rhi), (olo, ohi) = DRAFT_51[g["N"]]
        good = (rlo - 5e-5) <= g["rms"] <= (rhi + 5e-5) and (olo - 5e-4) <= g["on_grid"] <= (ohi + 5e-4)
        ok &= good
        print(f"  N={g['N']:<3} rms {g['rms']:.6f} in [{rlo}, {rhi}]  "
              f"on-grid {g['on_grid']:.4f} in [{olo}, {ohi}]  -> {'YES' if good else 'NO'}")
    print(f"  all groups consistent: {ok}")
    return ok


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    allow = frozenset(a for k, a in zip(sys.argv, sys.argv[1:]) if k == "--allow-missing")
    idx = run_index(allow)
    print(f"indexed {len(idx)} runs from result.json args")

    p3 = fig3_protocols()
    p4, used, zb = fig4_convergence(idx)
    p5, groups = fig5_grid_residual(idx)

    print(f"\n[fig4] protocol none, relu_qat, seed 0, zero baseline = {zb:.6f}")
    for u in used:
        print(f"  N={u['N']:<3} {u['protocol']:<11} best {u['best_mae']:.4f} @ep{u['best_epoch']:<3} "
              f"final {u['final_mae']:.4f}  {u['run']}")
    print("\n[fig5] after_snap, relu_qat, seed 0")
    for g in groups:
        print(f"  N={g['N']:<3} rel.RMS {g['rms']:.6f}  on-grid {g['on_grid']:.4f} "
              f"(from rho {g['on_grid_from_rho']:.4f})  tensors {g['n_tensors']} "
              f"elements {g['n_elements']}  rho==0 {g['frac_exact_zero']:.4f}  "
              f"median(nonzero) {g['median_nonzero']:.2e}  max {g['max']:.4f}  "
              f">0.45 {g['frac_gt_045']:.4f}")
        print(f"        {g['run']}  [{g['protocol']}]")
    if not report_51(groups):
        sys.exit("§5.1 校验不通过：图的区间与草稿不符，别把不一致的图当产物")

    from PIL import Image
    for p in (p3, p4, p5):
        with Image.open(p) as im:
            print(f"\nwrote {p}  {im.size[0]}x{im.size[1]} px  dpi={im.info.get('dpi')}")


if __name__ == "__main__":
    main()
