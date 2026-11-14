"""vendor8: FP32 baseline of one exported ONNX graph on the 200 test bins (no quantisation)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort

K = 200


def main() -> None:
    tag = sys.argv[1]
    onnx_path = ROOT / "npu_quant" / "onnx" / f"{tag}.onnx"
    meta = json.loads((ROOT / "npu_quant" / "testbin" / "meta.json").read_text())
    targets = np.array(meta["targets"], dtype=np.float64)[:K]
    bins = [ROOT / "npu_quant" / "testbin" / f"{i:04d}.bin" for i in range(K)]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    t0 = time.time()
    preds = np.empty(K)
    for i, b in enumerate(bins):
        x = np.fromfile(b, dtype=np.float32).reshape(1, 1, 3, 224, 224)
        preds[i] = float(sess.run(None, {iname: x})[0].ravel()[0])
    err = np.abs(preds - targets)
    res = {
        "kind": "fp32_onnx_x86_run", "tag": tag, "onnx": str(onnx_path.relative_to(ROOT)),
        "n": K, "mae": float(err.mean()),
        "rmse": float(np.sqrt(((preds - targets) ** 2).mean())),
        "max_abs_err": float(err.max()), "elapsed_s": round(time.time() - t0, 1),
        "preds": [float(v) for v in preds],
    }
    out = ROOT / "npu_quant" / "out" / f"vendor8_fp32_{tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"{tag}: FP32 MAE={res['mae']:.6f} n={K} elapsed={res['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
