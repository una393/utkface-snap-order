"""第 20 轮补跑的下游测量：新 run 的部署读数写进新表，历史表逐字节不动。

为什么不动 quant_runs/deploy_table_v2.csv：
  scripts/paper2_figs.py 断言主配置（γ=0.99 且 split_seed=0）恰好 23 行，
  scripts/heldout_eval.py 按 (mode,N,seed,decay,split) 建键并做聚合（pick() 要求键唯一）。
  本轮 A 组要把 seed 补到 8 个，直接往 v2 追加会让这两个脚本当场失败。
  所以新行写 quant_runs/deploy_table_v3.csv，v2 只做备份与校验（md5 进报告）。

用法：
  python scripts/round20_eval.py --dump     # 备份 v2 + 跑 deploy_table.py 落 v3 与逐图 npz
  python scripts/round20_eval.py --check    # 只校验：v2 未被改动、v3 含全部历史行
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PY = r"C:/Users/24399/.conda/envs/e2e/python.exe"
ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "quant_runs" / "deploy_table_v2.csv"
BAK = ROOT / "quant_runs" / "_bak_deploy_table_v2.csv"
V3 = ROOT / "quant_runs" / "deploy_table_v3.csv"
PERI = ROOT / "quant_runs" / "perimage"
REPORT = ROOT / "quant_runs" / "round20_deploy_check.json"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else ""


def backup() -> None:
    if V2.exists() and not BAK.exists():
        shutil.copy2(V2, BAK)
        print(f"backup {V2.name} -> {BAK.name} md5={md5(BAK)}", flush=True)
    print(f"v2 md5={md5(V2)}  bak md5={md5(BAK)}", flush=True)


def dump() -> None:
    backup()
    cmd = [PY, "scripts/deploy_table.py", "--out", str(V3.relative_to(ROOT)).replace("\\", "/"),
           "--dump", "quant_runs/perimage"]
    print("run:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT))
    print("deploy_table rc=", p.returncode, flush=True)
    if p.returncode != 0:
        sys.exit(p.returncode)
    check()


def check() -> None:
    """历史行必须逐字节保留：v2 的每一行在 v3 里出现且只出现一次。"""
    v2 = [l for l in V2.read_text(encoding="utf-8").splitlines() if l.strip()]
    v3 = [l for l in V3.read_text(encoding="utf-8").splitlines() if l.strip()] if V3.exists() else []
    missing = [l for l in v2 if v3.count(l) != 1]
    new = [l for l in v3 if l not in v2]
    rep = {"v2_md5": md5(V2), "bak_md5": md5(BAK), "v3_md5": md5(V3),
           "v2_rows": len(v2) - 1, "v3_rows": len(v3) - 1, "new_rows": len(new),
           "v2_untouched": md5(V2) == md5(BAK) if BAK.exists() else None,
           "history_rows_missing_or_duplicated": len(missing)}
    REPORT.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2), flush=True)
    for l in missing[:5]:
        print("MISSING:", l, flush=True)
    if missing:
        print(f"FAIL: {len(missing)} 行历史记录在 v3 里缺失或被改写", flush=True)
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    PERI.mkdir(parents=True, exist_ok=True)
    if a.dump:
        dump()
    else:
        check()


if __name__ == "__main__":
    main()
