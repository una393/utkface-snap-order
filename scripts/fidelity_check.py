"""Algorithm 1 as a single entry point: may a deployment fidelity score be quoted?

    python scripts/fidelity_check.py --preset demo --out quant_runs/fidelity_check.json
    python scripts/fidelity_check.py --run results/utk-relu_qat-none-s0-1793030915 --key model_state_dict

Input: an exported checkpoint and the deployment quantizer, described on the command
line.  Output: the residual of the exported tensor under that quantizer, whether the
quantizer's clipping bound binds, whether the tensor already sits on the deployment
grid, the centered cosine, and the branch of Algorithm 1 the model falls in -- drop the
fidelity metric, quote it with its provenance, or report the residual instead of a
cosine.

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
OUT_DEFAULT = ROOT / "quant_runs" / "fidelity_check.json"
TAU = 5e-3            # Proposition 1 / Algorithm 1 tolerance: 35 % above the largest
                      # residual measured in the paper (3.7e-3, section 5.1)
ON_GRID = 0.05        # a weight counts as on-grid within this fraction of a grid step

# Reproducible case list behind quant_runs/fidelity_check.json.
PRESET = {
    "demo": [
        dict(label="live@1", run="results/utk-relu_qat-none-s1-1793445602", key="model_state_dict"),
        dict(label="live@20", run="results/utk-relu_qat-none-s0-1793030915", key="model_state_dict"),
        dict(label="ema_before@20", run="results/utk-relu_qat-before_snap-s0-1793032011", key="ema_state_dict"),
        dict(label="ema_after@20", run="results/utk-relu_qat-after_snap-s0-1793033217", key="ema_state_dict"),
        dict(label="projection off", run="results/utk-relu_qat-none-s0-1793111850", key="model_state_dict"),
        dict(label="4 bit export of the 8 bit projection", run="results/utk-relu_qat-before_snap-s0-1793032011",
             key="ema_state_dict", bits=4, training_bits=8),
    ],
}


def load_weights(path: Path, key: str) -> dict:
    """Weight tensors of a checkpoint, selected as the rest of the repository selects them."""
    ckpt = torch.load(path / "best.pth" if path.is_dir() else path,
                      map_location="cpu", weights_only=False)
    sd = ckpt[key]
    return {k: v.float() for k, v in sd.items() if k.endswith(".weight") and v.ndim >= 2}


def quantize(flat: torch.Tensor, bits: int, rounding: str, scale: torch.Tensor | None = None):
    """Deployment quantizer on a [rows, -1] view: absmax scale per row, RTN, clamp."""
    qmax = 2 ** (bits - 1) - 1
    if scale is None:
        scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    x = flat / scale
    q = x.round() if rounding == "rtn" else torch.floor(x + torch.rand_like(x))
    return q.clamp(-qmax - 1, qmax).mul(scale), scale


def mse_scale(flat: torch.Tensor, bits: int) -> torch.Tensor:
    """MSE-optimal step per output channel, i.e. an AdaRound / BRECQ style training rule."""
    qmax = 2 ** (bits - 1) - 1
    amax = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    best = amax / qmax
    best_err = torch.full_like(amax, float("inf"))
    for f in torch.linspace(0.30, 1.0, 29):
        s = amax * f / qmax
        err = ((flat / s).round().clamp(-qmax - 1, qmax).mul(s) - flat).pow(2).mean(dim=1, keepdim=True)
        take = err < best_err
        best, best_err = torch.where(take, s, best), torch.where(take, err, best_err)
    return best


def cos_centered(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
    na, nb = a.norm(), b.norm()
    return float(a @ b / (na * nb)) if float(na * nb) else 1.0


def check_tensor(w: torch.Tensor, spec) -> dict:
    """One tensor: residual, clipping, on-grid share, centered cosine, step comparison."""
    bits, qmax = spec.bits, 2 ** (spec.bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    absmax = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    wq, s = quantize(flat, bits, spec.deployment_rounding,
                     None if spec.scale_mode == "recomputed" else absmax)
    resid = (wq - w.reshape(flat.shape)).abs() / s     # in grid steps, per output channel
    out = {
        "rho": float(((wq - flat).pow(2).sum() / flat.pow(2).sum()).sqrt()),
        "cos_centered": cos_centered(flat, wq),
        "on_grid_frac": float((resid < ON_GRID).float().mean()),
        "max_resid_steps": float(resid.max()),
        "clip_binds": bool((flat.abs() / s > qmax + 1e-9).any()),
        "num": float((wq - flat).pow(2).sum()),
        "den": float(flat.pow(2).sum()),
    }
    if spec.training_rule == "mse-optimal":
        s2 = mse_scale(flat, bits)
        wq2, _ = quantize(flat, bits, "rtn", s2)
        out["rho_under_training_step"] = float(((wq2 - flat).pow(2).sum() / flat.pow(2).sum()).sqrt())
    return out


def branch(rho: float, spec) -> tuple[str, str]:
    """Algorithm 1's three returns, in the order the algorithm tests them."""
    if (spec.training_rule != "same" or spec.deployment_rounding != "rtn"
            or spec.bits != spec.training_bits):
        return "report-rho-only", "the deployment quantizer differs from the training projection"
    if rho <= spec.tau:
        return "drop-metric", "the export step reproduces the tensor within tau"
    return "quote-with-provenance", "residual exceeds tau under the same rule and step"


ADVICE = {
    "drop-metric": "Report task MAE after export quantization: the fidelity score is not quotable, and Proposition 1 covers this case.",
    "report-rho-only": "Report the residual under the deployment step: a centered cosine is not comparable across steps or rules.",
    "quote-with-provenance": "Quote the centered cosine with its interval, bit width and evaluation basis; it cannot rank models.",
}


def resolve(path_like) -> Path:
    """Preset paths are relative to the repository root; fall back to the working
    directory so the installed entry point works inside a checkout."""
    for base in (ROOT, Path.cwd()):
        cand = base / path_like
        if cand.exists():
            return cand
    raise FileNotFoundError(ROOT / path_like)


def run_case(spec) -> dict:
    path = resolve(spec.run)
    weights = load_weights(path, spec.key)
    rows = [dict(check_tensor(w, spec), name=n) for n, w in weights.items()]
    den = sum(r["den"] for r in rows)
    rho = (sum(r["num"] for r in rows) / den) ** 0.5 if den else 0.0
    worst = max(rows, key=lambda r: r["rho"])
    verdict, reason = branch(rho, spec)
    sweep = {str(t): branch(rho, argparse.Namespace(**{**vars(spec), "tau": t}))[0]
             for t in (1e-3, TAU, 1e-2)}
    return {
        "label": spec.label, "run": spec.run, "key": spec.key,
        "deployment_bits": spec.bits, "training_bits": spec.training_bits,
        "tensors": len(rows), "rho": rho,
        "cos_centered_min": min(r["cos_centered"] for r in rows),
        "on_grid_frac_min": min(r["on_grid_frac"] for r in rows),
        "max_resid_steps": max(r["max_resid_steps"] for r in rows),
        "clip_binds": any(r["clip_binds"] for r in rows),
        "worst_tensor": {"name": worst["name"], "rho": worst["rho"]},
        "verdict": verdict, "reason": reason, "advice": ADVICE[verdict],
        "verdict_at_other_tau": sweep,
    }


def shadow_sweep(spec, modes=("before_snap", "after_snap"), interval=20, seeds=range(8)) -> dict:
    """Largest residual over the shadow checkpoints at one interval, the number tau is set from."""
    picked = {}
    for d in sorted(resolve("results").glob("utk-relu_qat-*")):
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
    rows = []
    for mode in modes:
        for seed in seeds:
            d = picked.get((mode, seed))
            if d is None:
                continue
            weights = load_weights(d, "ema_state_dict")
            per = [check_tensor(w, spec) for w in weights.values()]
            rows.append({"run": d.name, "mode": mode, "seed": seed,
                         "rho": (sum(r["num"] for r in per) / sum(r["den"] for r in per)) ** 0.5})
    rhos = [r["rho"] for r in rows]
    return {"interval": interval, "checkpoints": len(rows),
            "rho_min": min(rhos), "rho_max": max(rhos), "rows": rows}


def build_specs(args) -> list:
    if args.run:
        specs = [argparse.Namespace(label=Path(p).name, run=p, key=args.key,
                                    bits=args.bits, training_bits=args.training_bits)
                 for p in args.run]
    else:
        specs = []
        for case in PRESET[args.preset]:
            merged = dict(run="", key=args.key, bits=args.bits, training_bits=None)
            merged.update(case)
            merged["training_bits"] = merged["training_bits"] or merged["bits"]
            specs.append(argparse.Namespace(**merged))
    for s in specs:
        s.tau, s.deployment_rounding = args.tau, args.deployment_rounding
        s.training_rule, s.scale_mode = args.training_rule, args.scale_mode
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESET), default="demo")
    ap.add_argument("--run", action="append", default=[], help="checkpoint directory or .pth file")
    ap.add_argument("--key", default="model_state_dict", help="checkpoint key holding the exported weights")
    ap.add_argument("--bits", type=int, default=8, help="deployment bit width")
    ap.add_argument("--training-bits", type=int, default=None, help="bit width of the training projection")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--deployment-rounding", choices=("rtn", "stochastic"), default="rtn")
    ap.add_argument("--training-rule", choices=("same", "mse-optimal"), default="same")
    ap.add_argument("--scale-mode", choices=("recomputed", "fixed-amax"), default="recomputed")
    ap.add_argument("--sweep-seeds", action="store_true", help="add the across-seed shadow residual sweep")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    if args.training_bits is None:
        args.training_bits = args.bits

    torch.manual_seed(0)
    specs = build_specs(args)
    results = [run_case(s) for s in specs]
    report = {"tau": args.tau, "on_grid_tolerance": ON_GRID,
              "rho": "relative RMS residual ||W - S(W)|| / ||W|| over all projected tensors, eq. (4)",
              "deployment": {"rounding": args.deployment_rounding, "scale": args.scale_mode},
              "training_rule": args.training_rule, "cases": results}
    if args.sweep_seeds:
        report["shadow_sweep"] = shadow_sweep(specs[0])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + chr(10), encoding="utf-8")
    for r in results:
        print("%-40s %d/%d bit  rho %.2e  cos %.6f  on grid %.4f  -> %s"
              % (r["label"], r["deployment_bits"], r["training_bits"], r["rho"],
                 r["cos_centered_min"], r["on_grid_frac_min"], r["verdict"]))
    if "shadow_sweep" in report:
        sweep = report["shadow_sweep"]
        print("shadow residual at interval %d: %.2e - %.2e over %d checkpoints"
              % (sweep["interval"], sweep["rho_min"], sweep["rho_max"], sweep["checkpoints"]))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())