"""int8 量化测量链路（不依赖厂商工具链）。

对每个 checkpoint：
  1. 导出 ONNX（FP32）
  2. 统计结构性算子（Erf / Exp / Softmax / Pow —— 需要 LUT 或超越函数的那些）
  3. onnxruntime 静态量化（QDQ，int8）
  4. 在同一批测试样本上比 FP32 vs int8 的输出余弦、MAE、最大偏差
  5. 记录两者延迟（CPU，作为相对参考，不是 NPU 数字）

用法：
  python scripts/quant_eval.py --ckpt results/<run>/best.pth --head relu_qat --name H2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from data.base import open_raw
from data.transforms import SIZE
from data.utkface import scan, split_random
from models.net import DrivingNet

SUSPECT_OPS = ("Erf", "Exp", "Softmax", "Pow", "Tanh", "Sigmoid", "Sqrt", "Div")


class Reader(CalibrationDataReader):
    def __init__(self, arr, idx, name, n=200):
        self.items = []
        for i in idx[:n]:
            x = (np.asarray(arr[i], dtype="float32") / 255.0)
            x = (x - np.array([0.485, 0.456, 0.406], dtype="float32")) / \
                np.array([0.229, 0.224, 0.225], dtype="float32")
            self.items.append({name: x.transpose(2, 0, 1)[None, None].astype(np.float32)})
        self.i = 0

    def get_next(self):
        if self.i >= len(self.items):
            return None
        self.i += 1
        return self.items[self.i - 1]

    def rewind(self):
        self.i = 0


def count_ops(path: Path, scope: str | None = None) -> dict[str, int]:
    """scope='head' 时只统计控制头子图（按 ONNX 节点名里的作用域过滤）。"""
    m = onnx.load(str(path))
    out: dict[str, int] = {}
    for n in m.graph.node:
        if scope and scope not in n.name:
            continue
        out[n.op_type] = out.get(n.op_type, 0) + 1
    return out


def run(sess, x):
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x})[0].reshape(-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--weights", choices=["live", "ema"], default="live")
    p.add_argument("--raw", default="data/utkface_raw224.npy")
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--calib", type=int, default=200)
    p.add_argument("--test", type=int, default=200)
    p.add_argument("--out", default="quant_runs")
    a = p.parse_args()

    name = a.name or Path(a.ckpt).parent.name
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu")
    key = "ema_state_dict" if a.weights == "ema" else "model_state_dict"
    sd = ck.get(key) or ck.get("ema_state_dict") or ck.get("model_state_dict")
    print(f"[{name}] weights={a.weights} val_mae_live={ck.get('val_mae_live')} "
          f"val_mae_ema={ck.get('val_mae_ema')}", flush=True)

    net = DrivingNet(head=a.head, grid=(14, 14), seq=1)
    net.load_state_dict(sd, strict=False)
    net.eval()

    fp32_path = out / "model_fp32.onnx"
    torch.onnx.export(net, torch.zeros(1, 1, 3, SIZE[1], SIZE[0]), str(fp32_path),
                      opset_version=18, input_names=["input"], output_names=["out"],
                      dynamo=False)
    ops = count_ops(fp32_path)
    ops_head = count_ops(fp32_path, scope="head")
    suspect = {k: v for k, v in ops.items() if k in SUSPECT_OPS}
    suspect_head = {k: v for k, v in ops_head.items() if k in SUSPECT_OPS}
    print(f"[{name}] ops total={sum(ops.values())} suspect={suspect}", flush=True)
    print(f"[{name}] HEAD ops={sum(ops_head.values())} suspect={suspect_head or '{}'}",
          flush=True)

    recs = scan(a.data)
    _, val = split_random(recs, 0.15, 0)
    arr = np.load(a.raw, mmap_mode="r")
    calib_idx = [int(r["row"]) for r in val[: a.calib]]
    test_idx = [int(r["row"]) for r in val[a.calib: a.calib + a.test]]

    int8_path = out / "model_int8.onnx"
    quantize_static(str(fp32_path), str(int8_path), Reader(arr, calib_idx, "input", a.calib),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)

    so32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    so8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    def batch(idx):
        xs = []
        for i in idx:
            x = np.asarray(arr[i], dtype="float32") / 255.0
            x = (x - np.array([0.485, 0.456, 0.406], dtype="float32")) / \
                np.array([0.229, 0.224, 0.225], dtype="float32")
            xs.append(x.transpose(2, 0, 1)[None, None].astype(np.float32))
        return xs

    xs = batch(test_idx)
    y32, y8 = [], []
    t32 = t8 = 0.0
    for x in xs:
        t0 = time.perf_counter(); y32.append(float(run(so32, x)[0])); t32 += time.perf_counter() - t0
        t0 = time.perf_counter(); y8.append(float(run(so8, x)[0])); t8 += time.perf_counter() - t0
    y32, y8 = np.array(y32), np.array(y8)
    cos = float(np.dot(y32, y8) / (np.linalg.norm(y32) * np.linalg.norm(y8) + 1e-12))
    # 输出不跨零时（如年龄恒正）余弦会被"都在均值附近"抬高，必须配中心化余弦与相对 L2
    c32, c8 = y32 - y32.mean(), y8 - y8.mean()
    cos_c = float(np.dot(c32, c8) / (np.linalg.norm(c32) * np.linalg.norm(c8) + 1e-12))
    rel_l2 = float(np.linalg.norm(y32 - y8) / (np.linalg.norm(y32) + 1e-12))
    res = {
        "name": name, "head": a.head, "weights": a.weights,
        "ops_total": int(sum(ops.values())), "suspect_ops": suspect,
        "n_test": len(xs),
        "cosine_fp32_vs_int8": round(cos, 6),
        "cosine_centered": round(cos_c, 6),
        "rel_l2": round(rel_l2, 6),
        "suspect_ops_head": suspect_head,
        "mae_int8_vs_fp32": round(float(np.abs(y32 - y8).mean()), 6),
        "max_abs_diff": round(float(np.abs(y32 - y8).max()), 6),
        "latency_ms_fp32": round(1000 * t32 / len(xs), 3),
        "latency_ms_int8": round(1000 * t8 / len(xs), 3),
    }
    (out / "quant_result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[{name}] cos={cos:.4f} cos_centered={cos_c:.4f} relL2={rel_l2:.4f} "
          f"mae={res['mae_int8_vs_fp32']:.4f} maxdiff={res['max_abs_diff']:.3f}", flush=True)


if __name__ == "__main__":
    main()
