"""训练曲线：读 results/<run>/history.csv 出四联图。

版式沿用 e2e/utils/plot_curves.py：2×2（损失 / 验证 MAE / 学习率 / 过拟合间隙），
dpi=150，标签英文（避免字体缺字形），并在 MAE 面板画零基线。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def load(history_csv: str | Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with open(history_csv, newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return cols


def plot_history(history_csv: str | Path, out_png: str | Path,
                 title: str = "Training Progress") -> Path:
    c = load(history_csv)
    ep = c["epoch"]
    tl, vl = c.get("train_loss", []), c.get("val_mae_live", [])
    ve = c.get("val_mae_ema", [])
    lr = c.get("lr", [])
    zero = c.get("zero_mae", [float("nan")] * len(ep))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{title} - Epoch {int(ep[-1]) if ep else 0}", fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    ax1.plot(ep, tl, "b-", lw=2, marker="o", ms=4, label="Train MSE")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE", fontsize=12)
    ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3); ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    if tl:
        ax1.text(0.02, 0.95, f"Current: {tl[-1]:.5f}", transform=ax1.transAxes,
                 fontsize=10, va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

    ax2 = axes[0, 1]
    ax2.plot(ep, vl, "g-", lw=2, marker="^", ms=4, label="val MAE (live)")
    if ve:
        ax2.plot(ep, ve, "m--", lw=2, marker="s", ms=4, label="val MAE (EMA)")
    if zero and not np.isnan(zero[0]):
        ax2.axhline(zero[0], color="gray", ls=":", lw=2,
                    label=f"zero baseline = {zero[0]:.4f}")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("MAE", fontsize=12)
    ax2.set_title("Validation MAE vs zero baseline", fontsize=14, fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    if vl:
        ax2.text(0.02, 0.95, f"live {vl[-1]:.4f}", transform=ax2.transAxes, fontsize=10,
                 va="top", bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.5))

    ax3 = axes[1, 0]
    if lr:
        ax3.plot(ep, lr, "m-", lw=2, marker="o", ms=4, label="Learning Rate")
        ax3.set_yscale("log")
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("LR", fontsize=12)
    ax3.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax3.legend(); ax3.grid(alpha=0.3); ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax4 = axes[1, 1]
    if ve and vl:
        gap = [e - l for e, l in zip(ve, vl)]
        ax4.plot(ep[:len(gap)], gap, "purple", lw=2, marker="d", ms=4, label="EMA - live")
        ax4.axhline(0, color="black", ls="--", alpha=0.3)
        ax4.fill_between(ep[:len(gap)], 0, gap, where=np.array(gap) > 0, color="red",
                         alpha=0.2, label="EMA worse")
        ax4.fill_between(ep[:len(gap)], 0, gap, where=np.array(gap) < 0, color="green",
                         alpha=0.2, label="EMA better")
        ax4.legend()
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel("MAE gap", fontsize=12)
    ax4.set_title("EMA vs live (warm-up monitor)", fontsize=14, fontweight="bold")
    ax4.grid(alpha=0.3); ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}", flush=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python plot_history.py <history.csv> [out.png]")
    src = sys.argv[1]
    plot_history(src, sys.argv[2] if len(sys.argv) > 2 else str(Path(src).with_name("curves.png")))
