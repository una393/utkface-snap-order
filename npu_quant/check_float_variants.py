"""比较 exported ONNX / original_float / optimized_float 三种浮点模型在同一输入上的输出。"""
import sys

import numpy as np
import onnxruntime as ort

paths = sys.argv[1:4]
bins = ["npu_quant/testbin/0000.bin", "npu_quant/testbin/0001.bin", "npu_quant/testbin/0002.bin"]
for p in paths:
    sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    vals = []
    for b in bins:
        x = np.fromfile(b, dtype=np.float32).reshape(1, 1, 3, 224, 224)
        vals.append(float(sess.run(None, {name: x})[0].ravel()[0]))
    print(f"{p.split('/')[-1]:<45} " + "  ".join(f"{v:.8f}" for v in vals))
print("nash runtime (measured)".ljust(45), "0.38341203  0.24886014  0.38027298")
