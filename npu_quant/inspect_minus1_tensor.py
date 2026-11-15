"""Q3：gelu_s0_N1 全局 min 余弦 = -1.000000 的那个节点到底是什么张量。

做法：在 _original_float_model.onnx 上把该中间张量加为图输出，用宿主 onnxruntime 跑 8 张
测试图，统计其形状/范数/极值；再用该节点自己的阈值做一次对称 int8 伪量化，看余弦是否
复现 −1.000000。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import onnx
import onnxruntime as ort

TAG, NODE_OUT = "gelu_s0_N1", "/temporal/layers.0/norm2/LayerNormalization_mean1_output"
P = f"npu_quant/out/{TAG}/{TAG}_original_float_model.onnx"
m = onnx.load(P)
names = {i.name for i in m.graph.value_info} | {o.name for o in m.graph.output}
print("张量在图里存在:", NODE_OUT in names, "| graph.value_info 数量:", len(m.graph.value_info))
t = [v for v in m.graph.value_info if v.name == NODE_OUT]
if t:
    print("声明形状:", [d.dim_value or d.dim_param for d in t[0].type.tensor_type.shape.dim])

out = m.graph.output.add()
out.name = NODE_OUT
onnx.save(m, "npu_quant/out/_probe_mean1.onnx")
sess = ort.InferenceSession("npu_quant/out/_probe_mean1.onnx", providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
print("输出名:", [o.name for o in sess.get_outputs()])

vals = []
for k in range(8):
    x = np.fromfile(f"npu_quant/testbin/{k:04d}.bin", dtype=np.float32).reshape(1, 1, 3, 224, 224)
    y = sess.run([NODE_OUT], {iname: x})[0]
    vals.append(y.reshape(-1))
    print(f"  sample {k}: shape={y.shape} value={np.array2string(y.reshape(-1)[:4], precision=6)}")

v = np.concatenate(vals)
print(f"\n汇总 n={v.size}: mean={v.mean():.6g} std={v.std():.6g} |x|max={np.abs(v).max():.6g} L2={np.linalg.norm(v):.6g}")
print("是否全部为零:", bool(np.all(v == 0)), "| 是否常数:", bool(np.all(v == v.flat[0])))

# 用该节点自己的阈值做对称 int8 伪量化（阈值来自 quant_info: thresholds[[3.958074],[]]）
thr = 3.958074
scale = thr / 127.0
q = np.clip(np.round(v / scale), -127, 127)
dq = q * scale
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
print(f"\n阈值 {thr} -> scale {scale:.8g}；伪量化后 |dq|max={np.abs(dq).max():.6g}")
print(f"cos(v, dq) = {cos(v, dq):.6f}   （编译器报 -1.000000）")
print(f"quantized 全零: {bool(np.all(q == 0))}  q 取值: {sorted(set(q.tolist()))[:8]}")
