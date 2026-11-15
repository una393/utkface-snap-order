"""容器内：用 hmct executor 跑校准模型，取 mean1 的 float 与 fake-quant 张量。"""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from hmct.ir import load_model
from hmct.executor.ort2 import ORTExecutor2

ORTExecutor2.get_support_devices = classmethod(lambda cls: ["cpu"])

base = "/temporal/layers.0/norm2/LayerNormalization_mean1_output"
outs = [base, base + "_calibrated"]
ex = ORTExecutor2(load_model("/work/npu_quant/out/_probe_cal.onnx"))
ex.create_session()
iname = ex.get_inputs()[0].name
print("input:", iname, "| outputs:", [o.name for o in ex.get_outputs()] if hasattr(ex, "get_outputs") else "n/a")
res = {k: [] for k in outs}
for i in range(8):
    x = np.fromfile(f"/work/npu_quant/testbin/{i:04d}.bin", dtype=np.float32).reshape(1, 1, 3, 224, 224)
    y = ex.inference({iname: x})
    for k in outs:
        res[k].append(np.asarray(y[k]).reshape(-1))
a = {k: np.concatenate(v) for k, v in res.items()}
np.savez("/work/npu_quant/out/probe_mean1.npz", **{k.replace("/", "_"): v for k, v in a.items()})
for k, v in a.items():
    print(f"{k}: n={v.size} mean={v.mean():.6g} std={v.std():.6g} |x|max={np.abs(v).max():.6g} L2={np.linalg.norm(v):.6g}")
f, q = a[outs[0]], a[outs[1]]
cos = float(f @ q / (np.linalg.norm(f) * np.linalg.norm(q) + 1e-30))
print(f"cos(float, fake-quant) = {cos:.6f}")
print("float 前 8 个值:", np.array2string(f[:8], precision=6))
print("quant 前 8 个值:", np.array2string(q[:8], precision=6))
