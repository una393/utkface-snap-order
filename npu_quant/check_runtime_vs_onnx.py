"""核对 nash runtime 输出与 FP32 ONNX 输出是否一致（同一 .bin 输入）。"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

onnx_path = sys.argv[1]
bins = sys.argv[2:5]
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
rt = [0.38341203, 0.24886014, 0.38027298]  # 运行时实测（0000/0001/0002）
for i, b in enumerate(bins):
    x = np.fromfile(b, dtype=np.float32).reshape(1, 1, 3, 224, 224)
    y = float(sess.run(None, {name: x})[0].ravel()[0])
    print(f"{Path(b).name}: onnx_fp32={y:.8f}  nash_runtime={rt[i]:.8f}  diff={y - rt[i]:+.2e}")
