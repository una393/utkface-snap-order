"""Graphical abstract: the five-rule protocol as a decision flow (Elsevier submission).
Writes drafts/figs/figGA.png at >=600 dpi. Numbers are the run-level readings of the paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DPI = 660
INK = "#1a1a1a"
ACC = "#2b5d8a"
WARN = "#8a3b2b"

fig, ax = plt.subplots(figsize=(8.4, 9.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 114)
ax.axis("off")


def box(x, y, w, h, text, fs=8.6, edge=INK, face="#ffffff", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.6,rounding_size=1.2",
                                linewidth=1.1, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, weight=weight, linespacing=1.35)


def arrow(x0, y0, x1, y1, label=None, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.1, color=INK,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, fontsize=8,
                color=INK, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


box(8, 98, 84, 12,
    "Weights exported from deployment-oriented QAT\n"
    "projection every $N$ steps, optional EMA shadow", fs=9.4, weight="bold")
arrow(50, 98, 50, 90)

box(8, 76, 84, 13,
    "Are the training projection and the deployment quantizer\n"
    "the same operator and rounding rule, nothing clipping,\n"
    "and the exported tensor already on the training grid?",
    fs=8.6, edge=ACC)

arrow(30, 76, 18, 66, "yes", lx=-1, ly=0.5)
arrow(70, 76, 82, 66, "no", lx=1, ly=0.5)

box(4, 52, 42, 13,
    "Fidelity scores are uninformative\n"
    "(centered cosine is 1.0000 by construction)\n"
    "Report task MAE after export quantization",
    fs=8.4, edge=WARN, face="#fbf3f1")
box(56, 52, 40, 13,
    "Report centered cosine, but with\n"
    "interval, bit width and accounting basis;\n"
    "never as a ranking between models",
    fs=8.4)

arrow(25, 52, 40, 43)
arrow(76, 52, 62, 43)

box(12, 30, 76, 12,
    "Five-rule decision table: change the knob that moves the outcome\n"
    "switch projection off (\u0394MAE 0.0185 \u2192 0.0047)  \u00b7  STE control: no cost\n"
    "if the overwrite is unavoidable, stochastic rounding removes 99.8 %",
    fs=8.3, edge=ACC, face="#f2f6fa")

arrow(50, 30, 50, 22)

box(12, 8, 76, 13,
    "Protocol exercised end to end on a target NPU compiler\n"
    "and on two open-source runtimes; every claim mapped\n"
    "to the artifact that regenerates it",
    fs=8.4, weight="bold")

fig.savefig("drafts/figs/figGA.png", dpi=DPI, bbox_inches="tight", facecolor="white")
print("wrote drafts/figs/figGA.png")


def landscape():
    """Elsevier asks for a 5 x 13 cm (2.6:1) graphical abstract. The portrait
    flow above turns into 3-4 pt type when scaled to 5 cm, so the same protocol
    is redrawn left to right at the submission aspect ratio with short labels."""
    f, a = plt.subplots(figsize=(13, 5))
    f.subplots_adjust(left=0, right=1, top=1, bottom=0)
    a.set_xlim(0, 130)
    a.set_ylim(0, 50)
    a.axis("off")

    def bx(x, y, w, h, text, fs=17.0, edge=INK, face="#ffffff", weight="normal"):
        a.add_patch(FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.5,rounding_size=1.6",
                                   linewidth=1.7, edgecolor=edge, facecolor=face))
        a.text(x + w / 2, y + h / 2, text, ha="center", va="center",
               fontsize=fs, color=INK, weight=weight, linespacing=1.30)

    def ar(x0, y0, x1, y1, label=None, lx=0, ly=0):
        a.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                    mutation_scale=22, linewidth=1.7, color=INK,
                                    shrinkA=0, shrinkB=0))
        if label:
            a.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, fontsize=16,
                   color=INK, ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

    bx(1, 10, 28, 34,
       "Export from\ndeployment-oriented\nQAT: project every\n$N$ steps, optional\nEMA shadow",
       fs=16.0, weight="bold")
    ar(29, 27, 35, 27)

    bx(36, 10, 36, 34,
       "Same operator and\nrounding rule, nothing\nclipping, tensor already\non the training grid?",
       fs=16.0, edge=ACC)

    ar(72.2, 36, 77.4, 38.5, "yes", lx=-1.2, ly=2.4)
    ar(72.2, 18, 77.4, 15.5, "no", lx=-1.2, ly=-2.4)

    bx(78, 27, 51, 17,
       "Fidelity score is 1.0000\nby construction:\nreport task MAE instead",
       fs=16.0, edge=WARN, face="#fbf3f1")
    bx(78, 9, 51, 16,
       "Report centered cosine\nwith interval, bit width\nand accounting basis",
       fs=16.0)

    a.add_patch(FancyBboxPatch((1, 1.0), 128, 7.4,
                               boxstyle="round,pad=0.4,rounding_size=1.4",
                               linewidth=1.4, edgecolor=ACC, facecolor="#f2f6fa"))
    a.text(65, 6.1,
           "Overwriting costs $+$0.0185 MAE at interval 1 (0.0047 at interval 20); "
           "the grid alone costs 0.0000",
           ha="center", va="center", fontsize=14.0, color=INK)
    a.text(65, 3.0,
           "stochastic rounding at an unchanged grid, interval and budget removes 99.8 %",
           ha="center", va="center", fontsize=14.0, color=INK)

    f.savefig("drafts/figs/figGA_landscape.png", dpi=DPI, bbox_inches=None,
              facecolor="white")
    print("wrote drafts/figs/figGA_landscape.png")


landscape()
