"""宿主侧 FP32 MAE：用导出的 ONNX 在同一 200 张测试 bin 上跑，作为编译前基线。"""
import glob
import json
import os

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta = json.load(open(os.path.join(ROOT, "npu_quant", "testbin", "meta.json")))
k = meta["k"]
targets = np.array(meta["targets"], dtype=np.float64)
bins = [os.path.join(ROOT, "npu_quant", "testbin", f"{i:04d}.bin") for i in range(k)]

results = {}
for path in sorted(glob.glob(os.path.join(ROOT, "npu_quant", "onnx", "*.onnx"))):
    tag = os.path.splitext(os.path.basename(path))[0]
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    preds = []
    for b in bins:
        x = np.fromfile(b, dtype=np.float32).reshape(1, 1, 3, 224, 224)
        preds.append(float(sess.run(None, {iname: x})[0].ravel()[0]))
    preds = np.array(preds)
    err = np.abs(preds - targets)
    results[tag] = {
        "kind": "fp32_onnx_x86_run",
        "n": k,
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(((preds - targets) ** 2).mean())),
        "max_abs_err": float(err.max()),
        "preds": [float(v) for v in preds],
    }
    print(f"{tag:<18} MAE={err.mean():.6f} RMSE={np.sqrt(((preds-targets)**2).mean()):.6f} max={err.max():.6f}", flush=True)

out = os.path.join(ROOT, "npu_quant", "out", "mae_fp32.json")
json.dump({"targets": [float(t) for t in targets], "n": k, "models": results}, open(out, "w"), indent=1)
print("wrote", out)
