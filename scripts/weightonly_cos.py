"""部署权重的"仅权重量化"输出余弦——篇 2 的因变量。

与 quant_eval.py 的区别：不量化激活、不经过 onnxruntime。
EMA 更新时机只改变权重本身，激活范围由权重决定、但 PTQ 校准的随机性会淹没信号
（quant_eval 的 relL2 在 0.79-0.85，几乎全是激活量化的错）。
这里只做 weight-only: 把部署权重按训练时的同一套方案（按输出通道对称）量化再反量化，
前向一次，与 FP32 输出比。位宽可变（8/4），所以能测"位宽越窄，更新时机的差异是否越大"。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn

from data.transforms import SIZE
from data.utkface import scan, split_random
from models.net import DrivingNet

MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")


def qdq_(net: nn.Module, bits: int) -> dict:
    """就地按输出通道对称量化再反量化；返回网格统计。"""
    qmax = 2 ** (bits - 1) - 1
    resid, ong = [], []
    with torch.no_grad():
        for m in net.modules():
            if not isinstance(m, (nn.Linear, nn.Conv2d)):
                continue
            w = m.weight.data
            flat = w.reshape(w.shape[0], -1)
            scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
            q = (flat / scale).round().clamp(-qmax - 1, qmax)
            r = (flat - q * scale).abs() / scale
            resid.append(r.flatten())
            ong.append((r < 0.05).float().flatten())
            m.weight.data = (q * scale).reshape(w.shape)
    r = torch.cat(resid)
    return {"resid_mean": float(r.mean()), "on_grid": float(torch.cat(ong).mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--weights", choices=["live", "ema"], default="ema")
    p.add_argument("--bits", default="8,4")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--raw", default="data/utkface_raw224.npy")
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--name", default="")
    a = p.parse_args()

    name = a.name or Path(a.ckpt).parent.name
    ck = torch.load(a.ckpt, map_location="cpu")
    key = "ema_state_dict" if a.weights == "ema" else "model_state_dict"
    sd = ck.get(key) or ck.get("ema_state_dict") or ck.get("model_state_dict")

    torch.manual_seed(0)
    net = DrivingNet(head=a.head, grid=(14, 14), seq=1)
    net.load_state_dict(sd, strict=False)
    net.eval()

    recs = scan(a.data)
    _, val = split_random(recs, 0.15, 0)
    arr = np.load(a.raw, mmap_mode="r")
    idx = [int(r["row"]) for r in val[: a.n]]

    xs = []
    for i in idx:
        x = np.asarray(arr[i], dtype="float32") / 255.0
        x = (x - MEAN) / STD
        xs.append(torch.from_numpy(x.transpose(2, 0, 1)[None, None].copy()))

    with torch.no_grad():
        yref = torch.cat([net(x).flatten() for x in xs]).numpy()
        base = {k: v.detach().clone() for k, v in net.state_dict().items()}
        rows = []
        for bits in [int(b) for b in a.bits.split(",")]:
            net.load_state_dict(base, strict=False)
            st = qdq_(net, bits)
            y = torch.cat([net(x).flatten() for x in xs]).numpy()
            cos = float(np.dot(yref, y) / (np.linalg.norm(yref) * np.linalg.norm(y) + 1e-12))
            yc, yrc = y - y.mean(), yref - yref.mean()
            cosc = float(np.dot(yrc, yc) / (np.linalg.norm(yrc) * np.linalg.norm(yc) + 1e-12))
            rel = float(np.linalg.norm(y - yref) / (np.linalg.norm(yref) + 1e-12))
            mae = float(np.abs(y - yref).mean())
            mx = float(np.abs(y - yref).max())
            rows.append({"bits": bits, "cos": round(cos, 4), "cos_centered": round(cosc, 4),
                         "relL2": round(rel, 4), "mae": round(mae, 4), "maxdiff": round(mx, 4),
                         "resid_mean": round(st["resid_mean"], 4),
                         "on_grid": round(st["on_grid"], 4)})
            print(f"[{name}] bits={bits} cos={cos:.4f} cos_c={cosc:.4f} relL2={rel:.4f} "
                  f"mae={mae:.4f} resid={st['resid_mean']:.4f} on_grid={st['on_grid']:.4f}",
                  flush=True)
    Path("quant_runs").mkdir(exist_ok=True)
    Path(f"quant_runs/wo_{name}.json").write_text(
        json.dumps({"name": name, "weights": a.weights, "rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
