"""篇 2 需要的事实一次算全：网格残差、影子-live 距离、吸附误差、冻结比例。

任何写进论文的数都从这里出，不手抄。

输出 quant_runs/paper2_facts.json + stdout 表。
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn

from models.net import DrivingNet


def quant_tensors(net: nn.Module) -> dict[str, torch.Tensor]:
    return {n: m.weight.detach().float()
            for n, m in net.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))}


def rel_l2(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    num = sum((a[k] - b[k]).pow(2).sum().item() for k in a)
    den = sum(b[k].pow(2).sum().item() for k in b)
    return (num / den) ** 0.5


def grid_stats(w: torch.Tensor, bits: int = 8) -> tuple[float, float]:
    """返回 (相对 RMS 吸附误差, 落在格点上的比例)。按输出通道对称。"""
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    q = (flat / scale).round().clamp(-qmax - 1, qmax)
    resid = (q * scale - flat).abs() / scale  # 以格距为单位
    num = (q * scale - flat).pow(2).sum().item()
    den = flat.pow(2).sum().item()
    return (num / den) ** 0.5 if den else 0.0, float((resid < 0.05).float().mean())


def load_run(d: Path, key: str) -> dict[str, torch.Tensor]:
    ck = torch.load(d / "best.pth", map_location="cpu")
    net = DrivingNet(head="relu_qat", grid=(14, 14), seq=1, pretrained=False)
    net.load_state_dict(ck[key], strict=False)
    return quant_tensors(net)


def hist_stats(d: Path) -> dict:
    """snap_err / frozen_frac 两列可能不存在（早期 run），缺失必须报 nan 而不是 0。"""
    rows = list(csv.DictReader(open(d / "history.csv")))
    ep = [r for r in rows if r.get("snap_err")]
    if not ep:
        return {"snap_err_mean": float("nan"), "frozen_frac_mean": float("nan"), "epochs": len(rows)}
    return {
        "snap_err_mean": sum(float(r["snap_err"]) for r in ep) / len(ep),
        "frozen_frac_mean": sum(float(r["frozen_frac"]) for r in ep) / len(ep),
        "epochs": len(rows),
    }


def require_main_decay(a: dict, run: str) -> float:
    """main() 的 glob 会选到 γ 补跑（同 (freq, mode, seed) 目录名分不出），只认主配置 γ=0.99。"""
    d = float(a["ema_decay"])
    if d != 0.99:
        raise ValueError("%s: ema_decay=%s != 0.99，glob 选到了 γ 补跑" % (run, d))
    return d


def main() -> None:
    facts: dict = {"runs": [], "distances": [], "by_freq": {}}
    table = {}
    for d in sorted(Path("results").glob("utk-relu_qat-*")):
        rj = d / "result.json"
        if not rj.exists():
            continue
        meta = json.loads(rj.read_text(encoding="utf-8"))
        a = meta["args"]
        require_main_decay(a, d.name)
        freq, mode, seed = a.get("snap_every", 1), a["ema_mode"], a.get("seed", 0)
        key = "ema_state_dict" if mode != "none" else "model_state_dict"
        w = load_run(d, key)
        num = sum(v.numel() for v in w.values())
        num_weights = sum(m.weight.numel() for m in
                          [m for m in DrivingNet(head="relu_qat", grid=(14, 14), seq=1,
                                                 pretrained=False).modules()
                           if isinstance(m, (nn.Linear, nn.Conv2d))])
        rms, on_grid = 0.0, 0.0
        for v in w.values():
            r, o = grid_stats(v)
            rms += r * v.numel() / num
            on_grid += o * v.numel() / num
        hs = hist_stats(d)
        rec = {"freq": freq, "mode": mode, "seed": seed, "run": d.name,
               "snap_err_rel_rms": hs["snap_err_mean"], "frozen_frac": hs["frozen_frac_mean"],
               "deployed_weights": num_weights,
               "grid_resid_rel_rms": rms, "on_grid_frac": on_grid,
               "mae_live": meta["val_mae_live"], "mae_ema": meta["val_mae_ema"]}
        facts["runs"].append(rec)
        table[(freq, mode, seed)] = w
        print(f"freq{freq:<3}{mode:<12}s{seed}  snap {rms:.4f}  on_grid {on_grid:.3f}  "
              f"snap_err {hs['snap_err_mean']:.2e}  frozen {hs['frozen_frac_mean']:.3f}", flush=True)

    live = {}
    for d in sorted(Path("results").glob("utk-relu_qat-*")):
        rj = d / "result.json"
        if not rj.exists():
            continue
        a = json.loads(rj.read_text(encoding="utf-8"))["args"]
        require_main_decay(a, d.name)
        live[(a.get("snap_every", 1), a["ema_mode"], a.get("seed", 0))] = \
            load_run(d, "model_state_dict")

    # 三类距离分开算，别混：① 同一 run 内 影子 vs live（EMA 到底改了多少）
    # ② 同 freq 同 seed 跨模式（更新时机造成的差异）③ 同配置跨 seed（run 间波动）
    buckets = {"shadow_vs_live": [], "across_order_same_seed": [], "across_seed": []}
    for k in table:
        f, m, s = k
        if m != "none" and k in live:
            buckets["shadow_vs_live"].append({"freq": f, "seed": s, "mode": m,
                                              "rel_l2": rel_l2(table[k], live[k])})
    for (f1, m1, s1), (f2, m2, s2) in itertools.combinations(table, 2):
        v = rel_l2(table[(f1, m1, s1)], table[(f2, m2, s2)])
        if f1 == f2 and s1 == s2 and m1 != m2:
            buckets["across_order_same_seed"].append({"freq": f1, "seed": s1,
                                                      "a": m1, "b": m2, "rel_l2": v})
            if {m1, m2} == {"before_snap", "after_snap"}:
                buckets.setdefault("before_vs_after", []).append(
                    {"freq": f1, "seed": s1, "rel_l2": v})
        elif f1 == f2 and m1 == m2 and s1 != s2:
            buckets["across_seed"].append({"freq": f1, "mode": m1, "seeds": f"{s1}-{s2}",
                                           "rel_l2": v})
    for name, vals in buckets.items():
        if not vals:
            continue
        xs = [x["rel_l2"] for x in vals]
        facts["distance_summary"] = facts.get("distance_summary", {})
        facts["distance_summary"][name] = {"n": len(xs), "min": min(xs),
                                           "median": sorted(xs)[len(xs) // 2], "max": max(xs)}
        print(f"{name:<22} n={len(xs):<3} min {min(xs):.4f}  med {sorted(xs)[len(xs)//2]:.4f}"
              f"  max {max(xs):.4f}", flush=True)
    facts["distances"] = buckets

    out = Path("quant_runs/paper2_facts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
