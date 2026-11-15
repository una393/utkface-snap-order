"""部署权重离 int8 网格有多远。

篇 2 的机理：吸附算子 Q 与平均算子 E 不可交换。
- after_snap：影子权重 = E[Q(w)]，是"网格点的平均"，网格每步在动 → 平均值落在网格外
- before_snap：影子权重 = Q(E[w])，导出前才吸附 → 恒在网格上
所以部署权重"离网格的距离"应该能区分两者，而验证 MAE 区分不了（已实测跨度 0.0004）。

指标：按输出通道的对称 int8 量化（scale = max|w|/127），
  residual = |w - round(w/scale)*scale| / scale   （单位：网格步长的倍数，0 = 恰好落在格点）
  on_grid  = residual < 0.05 的比例
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import torch


def stats_for(sd: dict, bits: int = 8) -> dict:
    qmax = 2 ** (bits - 1) - 1
    resid_all, on_grid, abserr = [], [], []
    per_tensor = {}
    for k, w in sd.items():
        if not torch.is_tensor(w) or not w.is_floating_point() or w.ndim < 2:
            continue
        flat = w.detach().float().reshape(w.shape[0], -1)
        scale = flat.abs().amax(dim=1, keepdim=True) / qmax
        scale = torch.clamp(scale, min=1e-12)
        q = torch.clamp(torch.round(flat / scale), -qmax, qmax)
        resid = (flat - q * scale).abs() / scale          # 网格步长单位
        per_tensor[k] = (float(resid.mean()), float((resid < 0.05).float().mean()))
        resid_all.append(resid.flatten())
        on_grid.append((resid < 0.05).float().flatten())
        abserr.append((flat - q * scale).abs().flatten())
    r = torch.cat(resid_all)
    return {
        "tensors": len(per_tensor),
        "residual_mean": float(r.mean()),
        "on_grid_frac": float(torch.cat(on_grid).mean()),
        "abs_err_mean": float(torch.cat(abserr).mean()),
        "worst_tensor": min(per_tensor.items(), key=lambda kv: kv[1][1])[0],
        "per_tensor": {k: {"resid": round(v[0], 4), "on_grid": round(v[1], 4)}
                       for k, v in list(per_tensor.items())[:6]},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--weights", choices=["live", "ema"], default="ema")
    p.add_argument("--name", default="")
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    key = "ema_state_dict" if a.weights == "ema" else "model_state_dict"
    sd = ck.get(key) or ck.get("ema_state_dict") or ck.get("model_state_dict")
    out = stats_for(sd)
    out.update({"name": a.name or Path(a.ckpt).parent.name, "weights": a.weights,
                "val_mae_live": ck.get("val_mae_live"), "val_mae_ema": ck.get("val_mae_ema")})
    print(json.dumps({k: v for k, v in out.items() if k != "per_tensor"},
                     indent=2, ensure_ascii=False))
    Path("quant_runs").mkdir(exist_ok=True)
    Path(f"quant_runs/{out['name']}_weights.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
