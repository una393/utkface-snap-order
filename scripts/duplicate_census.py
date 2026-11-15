
"""Census of exactly duplicated images in the corpus (byte-identical pixel arrays)."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.cwd()))
from data.utkface import scan, split_random
from scripts import heldout_eval as he

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
recs = scan("D:/data/utkface/UTKFace")
arr = np.load("data/utkface_raw224.npy", mmap_mode="r")
keys = {}
for i in range(len(recs)):
    k = hashlib.sha1(np.ascontiguousarray(arr[i]).tobytes()).hexdigest()
    keys.setdefault(k, []).append(i)
groups = [g for g in keys.values() if len(g) > 1]
dup_images = sum(len(g) - 1 for g in groups)
first, mon, held, val, _, _ = he.build_views(recs, 0)
val_rows = {r["row"] for r in val}
held_rows = [r["row"] for r in held]
dup_rows = {i for g in groups for i in g}
held_dup = sum(1 for r in held_rows if r in dup_rows)
held_dup_cross = 0
for g in groups:
    if any(i in val_rows for i in g) and any(i not in val_rows for i in g):
        held_dup_cross += sum(1 for i in g if i in val_rows)
out = {"n_images": len(recs), "n_duplicate_groups": len(groups), "n_redundant_copies": dup_images,
       "max_group": max((len(g) for g in groups), default=0),
       "held_images_in_any_dup_group": held_dup,
       "val_images_with_train_twin": held_dup_cross, "n_held": len(held)}
Path("quant_runs/dup_census.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
