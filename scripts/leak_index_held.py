
"""Index which held-out (external-subset) images have a near-duplicate on the
training side of the random split.

dHash finds candidates; a pixel comparison on the cached 224x224 arrays decides.
Writes quant_runs/split_leak_held.json.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.cwd()))
from data.utkface import scan, split_random
from scripts import heldout_eval as he

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SP = 0
recs = scan("D:/data/utkface/UTKFace")
arr = np.load("data/utkface_raw224.npy", mmap_mode="r")

first, mon, held, val, mon_names, inter = he.build_views(recs, SP)
assert len(held) == 1556 and len(val) == 3556, (len(held), len(val))

h = np.empty((len(recs), 64), dtype=np.uint8)
for i in range(len(recs)):
    im = Image.fromarray(np.asarray(arr[i])).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    h[i] = (a[:, 1:] > a[:, :-1]).ravel()
packed = np.packbits(h, axis=1)
pop = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)

rng = np.random.default_rng(0)
_val_rows_set = {r["row"] for r in val}
vi = rng.choice([r["row"] for r in val], 1500, replace=False)
ti = rng.choice([r["row"] for r in recs if r["row"] not in _val_rows_set], 1500, replace=False)
null = pop[np.bitwise_xor(packed[vi][:, None, :], packed[ti][None, :, :])].sum(2).ravel()

val_rows = np.array([r["row"] for r in val])
rows = np.empty(len(held), np.int64); ham = np.empty(len(held), np.int32); pix = np.full(len(held), np.nan)
for k, r in enumerate(held):
    v = r["row"]
    d = pop[np.bitwise_xor(packed, packed[v])].sum(1).astype(np.int16)
    d[val_rows] = 64
    j = int(d.argmin()); rows[k] = v; ham[k] = int(d[j])
    if ham[k] <= 8:
        a = np.asarray(arr[v], dtype=np.int16); b = np.asarray(arr[j], dtype=np.int16)
        pix[k] = float(np.abs(a - b).mean())

tol = np.isfinite(pix)
out = {
  "split_seed": SP, "n_held": len(held), "n_val": len(val),
  "null_hamming": {"median": float(np.median(null)), "p1": float(np.percentile(null, 1)),
                   "p5": float(np.percentile(null, 5))},
  "hamming": {"min": int(ham.min()), "median": float(np.median(ham)),
              "le4": int((ham <= 4).sum()), "le8": int((ham <= 8).sum())},
  "pixel_checked": int(tol.sum()),
  "pixel_meanabs": {"min": float(np.nanmin(pix)), "median": float(np.nanmedian(pix)) if tol.any() else None},
  "identical_lt1": int(np.nansum(pix < 1.0)),
  "near_lt4": int(np.nansum(pix < 4.0)),
  "held_rows": rows.tolist(), "hamming_per_image": ham.tolist(),
  "pixel_per_image": [None if not np.isfinite(v) else round(float(v), 4) for v in pix],
}
Path("quant_runs/split_leak_held.json").write_text(json.dumps(out), encoding="utf-8")
print("held", len(held), "hamming", out["hamming"], "identical<1:", out["identical_lt1"], "near<4:", out["near_lt4"])
