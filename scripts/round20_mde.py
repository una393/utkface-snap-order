"""等价检验的功效与最小可检测效应（只读 quant_runs/，写 quant_runs/mde_power.json）。

口径（与稿件 §5.3 一致）：
  sigma_d = 逐 seed 配对差的样本标准差（n = 8、df = 7）
  SE      = sigma_d / sqrt(n)
  MDE(80 %) = (t_{0.975,df} + t_{0.80,df}) * SE        # 非中心 t 的正态近似，df = 7
  临界 margin（p = 0.05 的最小界）= |d| + t_{0.975,df} * SE
  功效      = 两个单侧检验在 0.05 水平同时拒绝的概率（df = 7 的 t 分布）
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
DF = 7
ALPHA = 0.05
POWER = 0.80
OUT = ROOT / "quant_runs" / "mde_power.json"


def mde_block(d, se):
    t_two = stats.t.ppf(1 - ALPHA / 2, DF)   # 双侧 0.975 临界，用于 MDE 与区间
    t_one = stats.t.ppf(1 - ALPHA, DF)       # 单侧 0.95 临界，用于临界 margin 与功效
    t_power = stats.t.ppf(POWER, DF)
    mde = (t_two + t_power) * se
    crit = abs(d) + t_one * se
    power_at = {}
    for margin in (0.0005, 0.0010, 0.0016, 0.0020):
        power_at["%.4f" % margin] = float(
            stats.t.cdf(margin / se - t_one, DF) - stats.t.cdf(-margin / se + t_one, DF))
    return {
        "d": d, "se": se, "df": DF, "alpha": ALPHA,
        "t_crit_two_sided": t_two, "t_crit_one_sided": t_one, "t_power": t_power,
        "mde_80pct": mde, "critical_margin": crit,
        "power_at_margin": power_at,
        "sigma_d": se * math.sqrt(DF + 1),
    }


def main():
    dual = json.loads((ROOT / "quant_runs" / "dual_basis_report.json").read_text(encoding="utf-8"))
    runl = json.loads((ROOT / "quant_runs" / "runlevel2_stats.json").read_text(encoding="utf-8"))
    out = {"meta": {
        "script": "scripts/round20_mde.py",
        "formula": "MDE = (t(0.975,df) + t(0.80,df)) * SE, SE = sigma_d / sqrt(n)",
        "n_seeds": 8, "df": DF,
        "inputs": ["quant_runs/dual_basis_report.json", "quant_runs/runlevel2_stats.json"],
    }, "bases": {}}
    for basis, src in (("first400", dual["bases"]["first400"]),
                       ("heldout1556", dual["bases"]["heldout1556"])):
        cell = src["d"]
        blk = mde_block(cell["mean"], cell["se"])
        blk["sigma_d"] = cell["sd"]
        blk["n"] = cell["n"]
        out["bases"][basis] = blk
    tost = runl.get("tost_d", {})
    out["runlevel_tost"] = {
        "source": "quant_runs/runlevel2_stats.json:tost_d",
        "note": "runlevel2_stats 的 tost_d 记录逐 margin 的双单侧 p 值",
        "raw_keys": sorted(tost.keys()) if isinstance(tost, dict) else None,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for basis, blk in out["bases"].items():
        print("%-12s d=%.9f se=%.9f sigma_d=%.9f MDE=%.6f crit=%.6f power@0.0016=%.4f" % (
            basis, blk["d"], blk["se"], blk["sigma_d"], blk["mde_80pct"],
            blk["critical_margin"], blk["power_at_margin"]["0.0016"]))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
