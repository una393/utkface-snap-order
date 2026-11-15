"""两种平均顺序的导出权重在导出期 8 bit 再量化前后的距离（第十份评审 M3 的一次性测量）。

对同一 seed 的 before_snap 与 after_snap 两个 run 取 best.pth 的 ema_state_dict（before_snap 为未投影
平均、after_snap 为已投影平均），在"仅权重量化"的 Linear/Conv2d 权重上比较：
  d_raw = ||m_before - m_after|| / ||m_after||
  d_q8  = ||S8(m_before) - S8(m_after)|| / ||S8(m_after)||
S8 与 scripts/deploy_table.py:qdq_ 逐字同义（逐输出通道 absmax 对称、RTN、clamp 到 [-128, 127]）。
另报"再量化后仍然不同的权重占比"与跨 seed 参考距离（after_snap 的 seed 0 对 seed 1..7）。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLE = ROOT / "quant_runs" / "deploy_table_v4.csv"
OUT_JSON = ROOT / "quant_runs" / "order_requant.json"
OUT_MD = ROOT / "quant_runs" / "order_requant.md"
QMAX = 127


def pick_runs() -> dict:
    """主配置：间隔 20、γ = 0.99、20 轮、固定划分 split_seed = 0（None 在 train.py 里等价于 seed，
    seed 0 上两者相同）。每个 (mode, seed) 取唯一的这样一份 run。"""
    picked = {}
    for d in sorted(RESULTS.iterdir()):
        name = d.name
        if not (name.startswith("utk-relu_qat-before_snap-s") or
                name.startswith("utk-relu_qat-after_snap-s")):
            continue
        try:
            args = json.loads((d / "result.json").read_text(encoding="utf-8"))["args"]
        except Exception:
            continue
        mode, seed = args.get("ema_mode"), int(args.get("seed", -1))
        if args.get("snap_every") != 20 or float(args.get("ema_decay", 0)) != 0.99:
            continue
        if int(args.get("epochs", 0)) != 20:
            continue
        sp = args.get("split_seed")
        sp = seed if sp is None or int(sp) < 0 else int(sp)
        if sp != 0:
            continue
        prev = picked.get((mode, seed))
        if prev is None or args.get("split_seed") == 0:
            picked[(mode, seed)] = name
    return picked


def weights(run: str) -> dict:
    ck = torch.load(RESULTS / run / "best.pth", map_location="cpu", weights_only=False)
    sd = ck["ema_state_dict"]
    return {k: v.float() for k, v in sd.items() if k.endswith(".weight") and v.ndim >= 2}


def quant8(w: torch.Tensor) -> torch.Tensor:
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / QMAX
    q = (flat / scale).round().clamp(-QMAX - 1, QMAX)
    return (q * scale).reshape(w.shape)


def rel(a: dict, b: dict) -> float:
    num = sum(float(((a[k] - b[k]) ** 2).sum()) for k in b)
    den = sum(float((b[k] ** 2).sum()) for k in b)
    return (num / den) ** 0.5


def main() -> int:
    picked = pick_runs()
    seeds = sorted({s for (_, s) in picked})
    per_seed = []
    for seed in seeds:
        ra, rb = picked.get(("before_snap", seed)), picked.get(("after_snap", seed))
        assert ra and rb, (seed, ra, rb)
        wa, wb = weights(ra), weights(rb)
        assert set(wa) == set(wb)
        qa = {k: quant8(v) for k, v in wa.items()}
        qb = {k: quant8(v) for k, v in wb.items()}
        diff = sum(int((qa[k] != qb[k]).sum()) for k in qb)
        total = sum(int(v.numel()) for v in qb.values())
        per_seed.append(
            {
                "seed": seed,
                "before_run": ra,
                "after_run": rb,
                "d_raw": rel(wa, wb),
                "d_q8": rel(qa, qb),
                "frac_changed_after_q8": diff / total,
                "n_weights": total,
            }
        )

    # 跨 seed 参考：after_snap 内部两两的 d_raw 与 d_q8
    cross = []
    for i, s in enumerate(seeds):
        for t in seeds[i + 1 :]:
            wi = weights(picked[("after_snap", s)])
            wt = weights(picked[("after_snap", t)])
            cross.append(
                {
                    "pair": [s, t],
                    "d_raw": rel(wi, wt),
                    "d_q8": rel({k: quant8(v) for k, v in wi.items()},
                                {k: quant8(v) for k, v in wt.items()}),
                }
            )

    def mean(xs):
        return sum(xs) / len(xs)

    def sd(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    d_raw = [r["d_raw"] for r in per_seed]
    d_q8 = [r["d_q8"] for r in per_seed]
    summary = {
        "d_raw_mean": mean(d_raw),
        "d_raw_sd": sd(d_raw),
        "d_q8_mean": mean(d_q8),
        "d_q8_sd": sd(d_q8),
        "d_q8_median": sorted(d_q8)[len(d_q8) // 2],
        "frac_changed_after_q8_mean": mean([r["frac_changed_after_q8"] for r in per_seed]),
        "cross_seed_d_raw_mean": mean([c["d_raw"] for c in cross]),
        "cross_seed_d_q8_mean": mean([c["d_q8"] for c in cross]),
        "n_seeds": len(seeds),
        "ratio_q8_over_raw": mean(d_q8) / mean(d_raw),
    }
    OUT_JSON.write_text(
        json.dumps({"summary": summary, "per_seed": per_seed, "cross_seed": cross},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = ["# 导出期 8 bit 再量化前后：两种平均顺序的权重距离", "",
             f"- 对齐 seed：{summary['n_seeds']} 个（间隔 20、γ = 0.99、固定划分 split_seed = seed）",
             f"- 再量化前 d_raw = {summary['d_raw_mean']:.4f} ± {summary['d_raw_sd']:.4f}"
             f"（{100*summary['d_raw_mean']:.1f} %）",
             f"- 再量化后 d_q8 = {summary['d_q8_mean']:.5f} ± {summary['d_q8_sd']:.5f}"
             f"（{100*summary['d_q8_mean']:.2f} %），中位数 {summary['d_q8_median']:.5f}",
             f"- 再量化后仍不同的权重占比 {100*summary['frac_changed_after_q8_mean']:.2f} %",
             f"- 跨 seed 参考：d_raw = {summary['cross_seed_d_raw_mean']:.3f}、"
             f"d_q8 = {summary['cross_seed_d_q8_mean']:.3f}", "",
             "| seed | d_raw | d_q8 | 再量化后变化占比 |", "|---|---:|---:|---:|"]
    for r in per_seed:
        lines.append(f"| {r['seed']} | {r['d_raw']:.4f} | {r['d_q8']:.5f} | "
                     f"{100*r['frac_changed_after_q8']:.2f} % |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
