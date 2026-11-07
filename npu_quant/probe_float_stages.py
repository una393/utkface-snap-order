"""用 hmct executor 跑 original_float / optimized_float / calibrated / ptq 四个模型，定位 0.028 偏差出在哪一步。"""
import numpy as np

from hmct.executor.ort2 import ORTExecutor2
from hmct.ir import load_model

ORTExecutor2.get_support_devices = classmethod(lambda cls: ["cpu"])

base = "/work/npu_quant/out/relu_qat_s0_N1/relu_qat_s0_N1"
suffixes = ["_original_float_model.onnx", "_optimized_float_model.onnx", "_calibrated_model.onnx", "_ptq_model.onnx"]
bins = ["/work/npu_quant/testbin/0000.bin", "/work/npu_quant/testbin/0001.bin", "/work/npu_quant/testbin/0002.bin"]
ref_fp32 = [0.37035269, 0.22320983, 0.35101959]
ref_nash = [0.38341203, 0.24886014, 0.38027298]

xs = [np.fromfile(b, dtype=np.float32).reshape(1, 1, 3, 224, 224) for b in bins]
print("reference fp32(onnxruntime)  " + "  ".join(f"{v:.8f}" for v in ref_fp32))
print("reference nash runtime      " + "  ".join(f"{v:.8f}" for v in ref_nash), flush=True)
for suf in suffixes:
    ex = ORTExecutor2(load_model(base + suf))
    ex.create_session()
    iname = ex.get_inputs()[0].name
    oname = ex.get_outputs()[0].name
    vals = [float(np.ravel(ex.inference({iname: x})[oname])[0]) for x in xs]
    print(f"{suf:<28}".ljust(28) + "  ".join(f"{v:.8f}" for v in vals), flush=True)
