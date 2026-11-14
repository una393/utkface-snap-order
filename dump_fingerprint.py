"""每个 run 的实验指纹：没有它，论文里的数字和比赛材料一样不可回溯。"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "no-git"


def split_hash(records: list[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: (r["group"], r["idx"])):
        val = float(r.get("steer", r.get("target", 0.0)))
        h.update(f"{r['group']}:{r['idx']}:{val:.6f}".encode())
    return h.hexdigest()[:16]


def dump(out_dir: str | Path, cfg: dict, train: list[dict], val: list[dict]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"config": cfg, "git": git_hash(),
               "n_train": len(train), "n_val": len(val),
               "split_hash": split_hash(train + val)}
    p = out / "fingerprint.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return p
