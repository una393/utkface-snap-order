
import json, time
import numpy as np
from hbm_runtime import HB_HBMRuntime

ROOT = "/root/qat_test"
TAGS = ["relu_qat_s0_N1", "relu_qat_s0_N20", "gelu_s0_N1", "bilinear_glu_s0_N1"]
N = 200

meta = json.load(open(ROOT + "/testbin/meta.json"))
targets = np.asarray(meta["targets"], dtype=np.float64)[:N]
xs = [np.fromfile(ROOT + "/testbin/%04d.bin" % i, dtype=np.float32).reshape(1, 1, 3, 224, 224) for i in range(N)]

def first_arr(out):
    v = out
    while isinstance(v, dict):
        v = list(v.values())[0]
    return np.asarray(v)

res = {}
for tag in TAGS:
    rt = HB_HBMRuntime(ROOT + "/" + tag + ".hbm")
    if tag == TAGS[0]:
        print("runtime version:", rt.version, flush=True)
        print("  inputs:", rt.input_names, rt.input_shapes, rt.input_dtypes, flush=True)
        print("  outputs:", rt.output_names, rt.output_shapes, rt.output_dtypes, flush=True)
    for i in range(5):
        rt.run(xs[i])
    preds = np.empty(N); lat = np.empty(N)
    for i in range(N):
        t0 = time.perf_counter()
        out = rt.run(xs[i])
        lat[i] = (time.perf_counter() - t0) * 1e3
        preds[i] = float(first_arr(out).ravel()[0])
    err = np.abs(preds - targets)
    res[tag] = dict(mae=float(err.mean()), rmse=float(np.sqrt(((preds - targets) ** 2).mean())),
                    max_abs=float(err.max()), n=N,
                    lat_ms=dict(mean=float(lat.mean()), p50=float(np.percentile(lat, 50)),
                                p95=float(np.percentile(lat, 95)), p99=float(np.percentile(lat, 99)),
                                mn=float(lat.min()), mx=float(lat.max())),
                    preds=preds.tolist())
    r = res[tag]
    print(tag, "MAE %.6f RMSE %.6f max %.4f | lat mean %.3f p50 %.3f p95 %.3f p99 %.3f ms"
          % (r["mae"], r["rmse"], r["max_abs"], r["lat_ms"]["mean"], r["lat_ms"]["p50"], r["lat_ms"]["p95"], r["lat_ms"]["p99"]), flush=True)

json.dump(res, open(ROOT + "/board_eval.json", "w"), indent=1)
print("saved", ROOT + "/board_eval.json")
