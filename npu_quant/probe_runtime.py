"""测 hbrt4-run-model-nash 的单样本耗时（冷/热/dev-times），决定 MAE 评测的样本预算。"""
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np

LD = ":".join([
    "/usr/local/lib/python3.10/dist-packages/hbdk4/runtime/x86_64_unknown_linux_gnu/nash/lib",
    "/usr/ucp",
    "/usr/local/lib/python3.10/dist-packages/hbdk4/compiler/_mlir_libs",
])
HBM = sys.argv[1] if len(sys.argv) > 1 else "/work/npu_quant/out/relu_qat_s0_N1/relu_qat_s0_N1.hbm"
NAME = os.path.basename(HBM)[:-4]
ENV = dict(os.environ, LD_LIBRARY_PATH=LD)


def run(inp, outdir, extra=()):
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    p = subprocess.run(
        ["hbrt4-run-model-nash", "-f", HBM, "-n", NAME, "-i", inp, "-o", outdir, "--numpy", *extra],
        env=ENV, capture_output=True, text=True,
    )
    dt = time.time() - t0
    npys = sorted(glob.glob(os.path.join(outdir, "*.npy")))
    vals = [np.load(f).ravel()[0] for f in npys]
    return dt, p.returncode, vals


for tag, inp, extra in [
    ("A cold 0000", "/work/npu_quant/testbin/0000.bin", ()),
    ("B warm 0001", "/work/npu_quant/testbin/0001.bin", ()),
    ("C dev-times2 0000", "/work/npu_quant/testbin/0000.bin", ("-t", "2")),
    ("D 0002", "/work/npu_quant/testbin/0002.bin", ()),
]:
    dt, rc, vals = run(inp, "/tmp/probe_out", extra)
    print(f"{tag}: elapsed={dt:.2f}s rc={rc} out={vals}", flush=True)
