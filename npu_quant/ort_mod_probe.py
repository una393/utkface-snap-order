
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

def mk(op, inputs_and_vals):
    ins = [helper.make_tensor_value_info(n, helper.np_dtype_to_tensor_dtype(a.dtype), list(a.shape)) for n, a in inputs_and_vals]
    outs = [helper.make_tensor_value_info("y", helper.np_dtype_to_tensor_dtype(inputs_and_vals[0][1].dtype), None)]
    node = helper.make_node(op, [n for n, _ in inputs_and_vals], ["y"], name="n0")
    inits = [numpy_helper.from_array(a, n) for n, a in inputs_and_vals]
    g = helper.make_graph([node], "g", ins, outs, inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)], ir_version=8)

def try_session(tag, m):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 10
    so.use_deterministic_compute = True
    print(tag, "creating session ...", flush=True)
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    feed = {}
    for i in s.get_inputs():
        dt = np.int64 if "int64" in i.type else np.float32
        feed[i.name] = np.zeros(i.shape, dtype=dt)
    print(tag, "session OK; run ->", s.run(None, feed), flush=True)

try_session("MOD", mk("Mod", [("a", np.array([2], dtype=np.int64)), ("b", np.array([3], dtype=np.int64))]))
try_session("ADD", mk("Add", [("a", np.array([1.0,2.0,3.0,4.0], dtype=np.float32)), ("b", np.array([1.0]*4, dtype=np.float32))]))
print("DONE")
