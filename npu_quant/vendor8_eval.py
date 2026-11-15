"""vendor8 (container side): run one compiled .hbm on the x86 nash simulator, 200 test bins.

Same invocation as npu_quant/eval_mae_runtime.py (one hbrt4-run-model-nash process per image,
multiprocess pool), with per-index retry and per-run wall clock recorded.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = "/work"
RT_LIB = "/usr/local/lib/python3.10/dist-packages/hbdk4/runtime/x86_64_unknown_linux_gnu/nash/lib"
LD_PATH = ":".join([RT_LIB, "/usr/ucp",
                    "/usr/local/lib/python3.10/dist-packages/hbdk4/compiler/_mlir_libs"])
TAG = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 200
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 14
ATTEMPTS = 2
HBM = os.path.join(ROOT, "npu_quant", "out", TAG, TAG + ".hbm")
BINDIR = os.path.join(ROOT, "npu_quant", "testbin")
OUTDIR = tempfile.mkdtemp(prefix="rt_" + TAG + "_")


def run_one(idx: int):
    b = os.path.join(BINDIR, f"{idx:04d}.bin")
    err = ""
    for _ in range(ATTEMPTS):
        od = os.path.join(OUTDIR, f"{idx:04d}")
        os.makedirs(od, exist_ok=True)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = LD_PATH
        cmd = ["hbrt4-run-model-nash", "-f", HBM, "-n", TAG, "-i", b, "-o", od,
               "--numpy", "--log-level", "Error"]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        npy = os.path.join(od, "age.npy")
        if os.path.exists(npy):
            return idx, float(np.load(npy).ravel()[0]), None
        err = (p.stdout + p.stderr)[-400:]
    return idx, None, err


if __name__ == "__main__":
    meta = json.load(open(os.path.join(BINDIR, "meta.json")))
    targets = np.array(meta["targets"], dtype=np.float64)[:K]
    preds = np.full(K, np.nan)
    t0 = time.time()
    with mp.Pool(WORKERS) as pool:
        for idx, val, err in pool.imap_unordered(run_one, range(K)):
            if val is None:
                print(f"[fail] {idx}: {err}", flush=True)
            else:
                preds[idx] = val
    elapsed = time.time() - t0
    ok = ~np.isnan(preds)
    err_abs = np.abs(preds[ok] - targets[ok])
    res = {
        "kind": "hbm_nash_x86_runtime_measured", "tag": TAG, "hbm": HBM,
        "n_requested": K, "n_ok": int(ok.sum()), "n_fail": int((~ok).sum()),
        "mae": float(err_abs.mean()) if ok.any() else None,
        "rmse": float(np.sqrt(((preds[ok] - targets[ok]) ** 2).mean())) if ok.any() else None,
        "max_abs_err": float(err_abs.max()) if ok.any() else None,
        "workers": WORKERS, "elapsed_s": round(elapsed, 1),
        "preds": [None if np.isnan(v) else float(v) for v in preds],
    }
    out = os.path.join(ROOT, "npu_quant", "out", f"vendor8_int8_{TAG}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"{TAG}: n_ok={res['n_ok']}/{K} MAE={res['mae']} elapsed={res['elapsed_s']}s", flush=True)
