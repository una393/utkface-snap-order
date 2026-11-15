"""Resolution bound behind Proposition 1: how small must the export residual be
before a centered cosine reported to d decimals can move at all?

    python scripts/resolution_bound.py --out quant_runs/resolution_bound.json

For each checkpoint the exported tensor is pooled over all projected weight
matrices, the deployment quantizer recomputes its per-output-channel step, and the
script reports
    eps   = ||W_hat - W||_2 / ||W||_2          (relative RMS residual of eq. 4)
    kappa = ||W - mean(W)||_2 / ||W||_2        (how much centering shortens W)
    cos_c = centered cosine of eq. 17
    bound = 1 - 2 (eps/kappa)^2                (Proposition 1)
and whether 2 (eps/kappa)^2 <= p/2 for p = 10^-d, i.e. whether the printed value is
forced to 1.0000 no matter which checkpoint is scored.

The quantization convention is the one used everywhere else in this repository
(scripts/paper2_diagnostics.py:35, scripts/round20_order_requant.py:61): symmetric per
output channel, absmax scale, RTN, clamp to [-qmax-1, qmax].
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = ROOT / "quant_runs" / "resolution_bound.json"

PRESET = [
    dict(label="live@1", run="results/utk-relu_qat-none-s1-1793445602", key="model_state_dict"),
    dict(label="live@20", run="results/utk-relu_qat-none-s0-1793030915", key="model_state_dict"),
    dict(label="ema_before@20", run="results/utk-relu_qat-before_snap-s0-1793032011", key="ema_state_dict"),
    dict(label="ema_after@20", run="results/utk-relu_qat-after_snap-s0-1793033217", key="ema_state_dict"),
    dict(label="projection off", run="results/utk-relu_qat-none-s0-1793111850", key="model_state_dict"),
    dict(label="4 bit export", run="results/utk-relu_qat-before_snap-s0-1793032011",
         key="ema_state_dict", bits=4),
]


def quantize_rows(w: torch.Tensor, rows: int, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(rows, -1)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return (flat / scale).round().clamp(-qmax - 1, qmax).mul(scale).reshape(-1)


def analyse(rec, decimals: int) -> dict:
    bits = rec.get("bits", 8)
    ckpt = torch.load(ROOT / rec["run"] / "best.pth", map_location="cpu", weights_only=False)
    sd = ckpt[rec["key"]]
    tensors = [(k, v.float()) for k, v in sd.items() if k.endswith(".weight") and v.ndim >= 2]
    w = torch.cat([v.reshape(-1) for _, v in tensors])
    wq = torch.cat([quantize_rows(v, v.shape[0], bits) for _, v in tensors])
    eps = float((wq.double() - w.double()).norm() / w.double().norm())
    kappa = float((w.double() - w.double().mean()).norm() / w.double().norm())
    a, b = w.double() - w.double().mean(), wq.double() - wq.double().mean()
    cos = float(a @ b / (a.norm() * b.norm()))
    bound = 1.0 - 2.0 * (eps / kappa) ** 2
    p = 10.0 ** (-decimals)
    return dict(label=rec["label"], bits=bits, eps=eps, kappa=kappa, cos_centered=cos,
                bound=bound, slack=cos - bound,
                within_resolution=bool(2.0 * (eps / kappa) ** 2 <= p / 2),
                limit_eps=kappa * (p / 2) ** 0.5)


def sweep_checkpoints(interval: int = 20, seeds=range(8)) -> list:
    """Shadow checkpoints at one interval, the population tau_R has to cover."""
    picked = {}
    for d in sorted((ROOT / "results").glob("utk-relu_qat-*")):
        meta = d / "result.json"
        if not meta.exists():
            continue
        a = json.loads(meta.read_text(encoding="utf-8"))["args"]
        if a.get("ema_decay") != 0.99 or a.get("epochs") != 20 or a.get("snap_every") != interval:
            continue
        if a.get("quant_mode") == "ste" or "-ste" in d.name:
            continue
        split = a.get("split_seed")
        split = a.get("seed") if split is None or int(split) < 0 else int(split)
        if int(split) != 0:
            continue
        picked[(a.get("ema_mode"), int(a.get("seed", -1)))] = d
    out = []
    for mode in ("before_snap", "after_snap"):
        for seed in seeds:
            d = picked.get((mode, seed))
            if d is None:
                continue
            out.append(analyse(dict(label="%s@%d s%d" % (mode, interval, seed),
                                    run=str(d.relative_to(ROOT)), key="ema_state_dict"), 4))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--sweep-seeds", action="store_true", help="add the across-seed shadow sweep")
    a = ap.parse_args()
    rows = [analyse(r, a.decimals) for r in PRESET]
    sweep = sweep_checkpoints() if a.sweep_seeds else []
    print("%-18s %10s %8s %10s %10s %9s %s" % ("checkpoint", "eps", "kappa", "cos_c", "bound", "slack", "resolved"))
    for r in rows:
        print("%-18s %10.3e %8.4f %10.7f %10.7f %9.2e %s" %
              (r["label"], r["eps"], r["kappa"], r["cos_centered"], r["bound"], r["slack"],
               "yes" if r["within_resolution"] else "no"))
    worst = max(rows, key=lambda r: r["eps"])
    print("\nlargest eps %.3e (%s); smallest kappa %.4f" %
          (worst["eps"], worst["label"], min(r["kappa"] for r in rows)))
    print("resolution limit at %d decimals: eps <= kappa * %.4e (about %.2e with kappa = 1)"
          % (a.decimals, (10.0 ** (-a.decimals) / 2) ** 0.5, (10.0 ** (-a.decimals) / 2) ** 0.5))
    print("bound violated anywhere:", any(r["slack"] < 0 for r in rows))
    if sweep:
        print("shadow sweep: %d checkpoints, eps %.2e - %.2e, smallest slack %.2e, bound violated %s"
              % (len(sweep), min(r["eps"] for r in sweep), max(r["eps"] for r in sweep),
                 min(r["slack"] for r in sweep), any(r["slack"] < 0 for r in sweep)))
        print("all shadow checkpoints inside the four-decimal resolution:",
              all(r["within_resolution"] for r in sweep))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(decimals=a.decimals, rows=rows, shadow_sweep=sweep),
                                     indent=1) + "\n", encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
