"""导出 S100 编译用静态形状 ONNX（UTKFace 单帧 3x224x224，seq=1，grid=14x14）。

与仓库根目录 export_onnx.py 的区别：那个是驾驶工作区的 8 帧 / 224x384 口径，
这里按 train.py 里 UTKFace 实验的真实配置 DrivingNet(head=..., grid=(14,14), seq=1)。
不改训练代码，只读 checkpoint。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnx
import torch

from models.net import DrivingNet

SIZE = 224


def build(head: str, ckpt_path: str, weights: str = "ema") -> tuple[DrivingNet, dict]:
    net = DrivingNet(head=head, grid=(14, 14), seq=1, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "ema_state_dict" if weights == "ema" else "model_state_dict"
    sd = ckpt.get(key) or ckpt.get("ema_state_dict") or ckpt.get("model_state_dict")
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    net.eval()
    meta = {"ckpt": str(ckpt_path), "head": head, "weights": key,
            "val_mae_live": ckpt.get("val_mae_live"), "val_mae_ema": ckpt.get("val_mae_ema"),
            "train_args": {k: v for k, v in (ckpt.get("args") or {}).items()
                           if k in ("head", "ema_mode", "seed", "snap_every", "epochs")}}
    return net, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", choices=["ema", "live"], default="ema")
    a = p.parse_args()

    net, meta = build(a.head, a.ckpt, a.weights)
    n_par = sum(x.numel() for x in net.parameters())
    # nn.TransformerEncoder 在 torch>=2.x 会走 aten::_transformer_encoder_layer_fwd 融合快路，
    # 该算子无法导出到 ONNX（legacy exporter 直接报 UnsupportedOperatorError）。关掉快路即可退化成
    # 标准的 bmm/softmax 组合。
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    dummy = torch.zeros(1, 1, 3, SIZE, SIZE)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(net, dummy, a.out, opset_version=18,
                          input_names=["sequence_images"], output_names=["age"],
                          dynamo=False)
    print(f"[export] {a.out}  params={n_par/1e6:.3f}M  meta={json.dumps(meta)}")

    m = onnx.load(a.out)
    onnx.checker.check_model(m)
    for i in m.graph.input:
        dims = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"[onnx] input  {i.name} {dims}")
    for o in m.graph.output:
        dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"[onnx] output {o.name} {dims}")

    # ORT 与 torch 数值一致性
    import onnxruntime as ort
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(6):
        x = torch.from_numpy(rng.standard_normal((1, 1, 3, SIZE, SIZE), dtype=np.float32))
        with torch.no_grad():
            y_t = net(x).numpy().reshape(-1)
        y_o = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0].reshape(-1)
        worst = max(worst, float(np.abs(y_t - y_o).max()))
    print(f"[verify] onnxruntime vs torch max|diff| = {worst:.3e}")


if __name__ == "__main__":
    main()
