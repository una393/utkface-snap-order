import sys, onnxruntime as ort
sys.stdout.reconfigure(line_buffering=True)
from onnxruntime_extensions import get_library_path

MODEL = "/work/npu_quant/onnx/relu_qat_s0_N1.onnx"
HORIZON = "/usr/local/lib/python3.10/dist-packages/hmct/executor/horizon_operators.so"

def make(custom_libs, det=True, threads=10):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = threads
    so.use_deterministic_compute = det
    for p in custom_libs:
        so.register_custom_ops_library(p)
    return ort.InferenceSession(MODEL, sess_options=so, providers=["CPUExecutionProvider"])

for tag, libs, det in [("plain, no custom libs", [], True),
                       ("ext only", [get_library_path()], True),
                       ("horizon only", [HORIZON], True),
                       ("both", [get_library_path(), HORIZON], True)]:
    print("### try:", tag, flush=True)
    try:
        s = make(libs, det)
        import numpy as np
        y = s.run(None, {s.get_inputs()[0].name: np.zeros((1,1,3,224,224), np.float32)})[0]
        print("   OK", y.shape, float(y.reshape(-1)[0]), flush=True)
    except Exception as e:
        print("   EXC", type(e).__name__, str(e)[:300], flush=True)
print("done", flush=True)
