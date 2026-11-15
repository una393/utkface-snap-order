"""导出 ONNX（opset 18）。部署权重取 ema_state_dict，与比赛链路保持一致。"""
from __future__ import annotations

import argparse

import torch

from models.net import DrivingNet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="model.onnx")
    p.add_argument("--head", default="bilinear_glu")
    p.add_argument("--seq", type=int, default=8)
    args = p.parse_args()

    model = DrivingNet(head=args.head, seq=args.seq)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    dummy = torch.zeros(1, args.seq, 3, 224, 384)
    torch.onnx.export(model, dummy, args.out, opset_version=18,
                      input_names=["sequence_images"], output_names=["steer"],
                      dynamic_axes={"sequence_images": {0: "b"}, "steer": {0: "b"}})
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
