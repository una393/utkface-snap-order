"""在容器里跑 hmct 的 ptq（伪量化）模型，比较 fake-quant 模拟值 vs nash runtime 实测值。

用 hmct 的 ORTExecutor（它自动注册 horizon_operators.so + onnxruntime_extensions），
但把 device 强制成 cpu，避免本机 CUDA EP 建 session 段错误。
"""
import sys

import numpy as np

from hmct.executor.ort import ORTExecutorBase
from hmct.executor.ort2 import ORTExecutor2

ORTExecutor2.get_support_devices = classmethod(lambda cls: ["cpu"])

base = "/work/npu_quant/out/relu_qat_s0_N1/relu_qat_s0_N1"
bins = ["/work/npu_quant/testbin/0000.bin", "/work/npu_quant/testbin/0001.bin", "/work/npu_quant/testbin/0002.bin"]
rt = [0.38341203, 0.24886014, 0.38027298]

from hmct.ir import load_model  # noqa: E402

for tag, suffix in [("calibrated", "_calibrated_model.onnx"), ("ptq", "_ptq_model.onnx")]:
    try:
        model = load_model(base + suffix)
    except Exception as exc:  # noqa: BLE001
        print(tag, "load failed", exc)
        continue
    ex = ORTExecutor2(model)
    ex.create_session()
    print(tag, "inputs:", [i.name for i in ex.get_inputs()], "outputs:", [i.name for i in ex.get_outputs()])
    for i, b in enumerate(bins):
        x = np.fromfile(b, dtype=np.float32).reshape(1, 1, 3, 224, 224)
        y = ex.inference({ex.get_inputs()[0].name: x})[ex.get_outputs()[0].name]
        print(f"  {tag} {b.split('/')[-1]}: fake_quant={float(np.ravel(y)[0]):.8f} nash={rt[i]:.8f} diff={float(np.ravel(y)[0]) - rt[i]:+.2e}", flush=True)
