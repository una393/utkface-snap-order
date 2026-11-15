"""容器侧：hmct 伪量化模型（_ptq_model.onnx）在同一测试 bin 上的 MAE —— 这是「编译器模拟」。"""
import glob
import json
import os

import numpy as np

from hmct.executor.ort2 import ORTExecutor2
from hmct.ir import load_model

ORTExecutor2.get_support_devices = classmethod(lambda cls: ["cpu"])

ROOT = "/work"
meta = json.load(open(os.path.join(ROOT, "npu_quant", "testbin", "meta.json")))
k = meta["k"]
targets = np.array(meta["targets"], dtype=np.float64)
bins = [os.path.join(ROOT, "npu_quant", "testbin", f"{i:04d}.bin") for i in range(k)]
xs = [np.fromfile(b, dtype=np.float32).reshape(1, 1, 3, 224, 224) for b in bins]

results = {}
for path in sorted(glob.glob(os.path.join(ROOT, "npu_quant", "out", "*", "*_ptq_model.onnx"))):
    tag = os.path.basename(os.path.dirname(path))
    ex = ORTExecutor2(load_model(path))
    ex.create_session()
    iname = ex.get_inputs()[0].name
    oname = ex.get_outputs()[0].name
    preds = np.array([float(np.ravel(ex.inference({iname: x})[oname])[0]) for x in xs])
    err = np.abs(preds - targets)
    results[tag] = {
        "kind": "hmct_fakequant_simulated",
        "n": k,
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(((preds - targets) ** 2).mean())),
        "max_abs_err": float(err.max()),
        "preds": [float(v) for v in preds],
    }
    print(f"{tag:<20} MAE={err.mean():.6f} RMSE={np.sqrt(((preds-targets)**2).mean()):.6f} max={err.max():.6f}", flush=True)

out = os.path.join(ROOT, "npu_quant", "out", "mae_fakequant.json")
json.dump({"targets": [float(t) for t in targets], "n": k, "models": results}, open(out, "w"), indent=1)
print("wrote", out)
