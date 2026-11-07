
import onnx, numpy as np
from onnx import numpy_helper
m = onnx.load(r"C:\Users\24399\Desktop\paper-utkface\npu_quant\onnx\relu_qat_s0_N1.onnx")
g = m.graph
prod = {o: n for n in g.node for o in n.output}
target = "/temporal/layers.0/self_attn/Mod"
node = [n for n in g.node if n.name == target][0]
vals = []
for name in node.input:
    cn = prod[name]
    arr = numpy_helper.to_array(cn.attribute[0].t)
    vals.append(arr)
    print(name, cn.op_type, "dtype", arr.dtype, "shape", arr.shape, "val", arr.ravel()[:8])
a, b = vals
with np.errstate(all="ignore"):
    try:
        out = np.mod(a, b)
        print("np.mod ->", out, out.dtype)
    except Exception as e:
        print("np.mod failed:", type(e).__name__, e)
# who consumes Mod's output
users = [(n.op_type, n.name, i) for n in g.node for i, x in enumerate(n.input) if x == node.output[0]]
print("mod output name", node.output[0], "users:", users)
