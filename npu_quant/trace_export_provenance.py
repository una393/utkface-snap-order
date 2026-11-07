"""反查：每个导出的 ONNX 到底来自哪个 best.pth、用的是 ema_state_dict 还是 model_state_dict。

做法是穷举 results/*/best.pth，用同样构造的 DrivingNet 载入 ema / live 两套权重，
在与 ONNX 相同的固定输入上比较输出；最小误差的那一对就是来源。
"""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort
import torch

from models.net import DrivingNet

ONNX = sorted(glob.glob("npu_quant/onnx/*.onnx"))
x = np.fromfile("npu_quant/testbin/0000.bin", dtype=np.float32).reshape(1, 1, 3, 224, 224)
xt = torch.from_numpy(x)


def onnx_pred(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return float(s.run(None, {s.get_inputs()[0].name: x})[0].ravel()[0])


def net_pred(head, sd, key):
    net = DrivingNet(head=head, grid=(14, 14), seq=1, pretrained=False)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        return None, f"mismatch m={len(missing)} u={len(unexpected)}"
    net.eval()
    with torch.no_grad():
        return float(net(xt).reshape(-1)[0]), "ok"


refs = {p: onnx_pred(p) for p in ONNX}
print("onnx 参考输出:", {os.path.basename(k): round(v, 8) for k, v in refs.items()}, flush=True)

rows = []
for ck in sorted(glob.glob("results/*/best.pth")):
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    args = ckpt.get("args") or {}
    head = args.get("head") or os.path.basename(os.path.dirname(ck)).split("-")[1]
    rec = {
        "ckpt": os.path.dirname(ck).split(os.sep)[-1],
        "head": head,
        "ema_mode": args.get("ema_mode"),
        "seed": args.get("seed"),
        "snap_every": args.get("snap_every"),
        "has_ema": "ema_state_dict" in ckpt,
        "has_live": "model_state_dict" in ckpt,
        "val_live": ckpt.get("val_mae_live"),
        "val_ema": ckpt.get("val_mae_ema"),
    }
    for kind in ("ema", "live"):
        sd = ckpt.get("ema_state_dict" if kind == "ema" else "model_state_dict")
        if sd is None:
            rec[kind] = None
            continue
        pred, note = net_pred(head, sd, kind)
        rec[kind] = pred
        if pred is None:
            print("  !", rec["ckpt"], kind, note, flush=True)
    rows.append(rec)
    print(f"{rec['ckpt']:<45} ema={rec.get('ema')} live={rec.get('live')}", flush=True)

print("\n### 最佳匹配（|onnx - net| 最小）")
for p, v in refs.items():
    best = []
    for r in rows:
        for kind in ("ema", "live"):
            if r.get(kind) is not None:
                best.append((abs(r[kind] - v), r["ckpt"], kind, r[kind]))
    best.sort()
    print(f"\n{os.path.basename(p)}  onnx={v:.8f}")
    for d, ck, kind, pv in best[:3]:
        print(f"   |diff|={d:.2e}  {ck}  weights={kind}  net={pv:.8f}")

json.dump({"onnx_refs": refs, "runs": rows}, open("npu_quant/out/export_provenance.json", "w"), indent=1)
print("\nwrote npu_quant/out/export_provenance.json")
