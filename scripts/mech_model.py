"""Reduce the --log-mech runs to the mechanism quantities the paper reports.

Every projection event writes one line to mech.jsonl (train.py --log-mech).  The reference
point of the event is the weight vector as it stood after the previous projection, so the
logged update norm u is the drift accumulated since that projection and the norm after the
write s is the displacement the projection itself applied.  Their ratio s/u = 1 - eaten_global
is therefore read as "how far the overwrite moves the tensor, in units of the drift it was
reacting to", and proj_global is the fraction of that drift which survives along its own
direction.

The script prints, per interval, the mean of those quantities, the number of projections, and
the deployment cost of the same cells on the held-out basis, then compares two one-parameter
readings of the interval axis fitted on interval 1: a frequency reading (cost proportional to
the number of projections) and an erasure reading (cost proportional to the erased fraction
times the number of projections).  Both are out-of-sample at intervals 5 and 20.

    python scripts/mech_model.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MECH = ROOT / "results_mech"
HELDOUT = ROOT / "quant_runs" / "heldout_eval_v3.csv"
OUT_CSV = ROOT / "quant_runs" / "mech_model.csv"
OUT_JSON = ROOT / "quant_runs" / "mech_model.json"
INTERVALS = (1, 5, 20)
OFF = "0"  # N = 0 marks the never-projecting arm in the deployment table


def costs() -> dict[tuple[int, int], float]:
    """Per-seed deployment cost of the live arm: MAE at interval N minus MAE with no projection."""
    cell: dict[tuple[str, str, int], float] = {}
    with open(HELDOUT, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["mode"] != "none" or r["subset"] != "heldout_1556":
                continue
            cell[(r["N"], r["seed"])] = float(r["mae_fp32"])
    return {(int(n), int(s)): cell[(str(n), str(s))] - cell[(OFF, str(s))]
            for (n, s) in cell if n != OFF}


def read_run(d: Path) -> dict:
    event = [json.loads(l) for l in (d / "mech.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads((d / "result.json").read_text(encoding="utf-8"))["args"]
    mean = lambda xs: sum(xs) / len(xs)
    return {
        "run": d.name,
        "interval": int(meta["snap_every"]),
        "seed": int(meta["seed"]),
        "events": len(event),
        "kick_over_drift": mean([1.0 - e["eaten_global"] for e in event]),
        "proj_coef": mean([e["proj_global"] for e in event]),
        "u_over_delta_mean": mean([e["u_over_delta"]["mean"] for e in event]),
        "u_over_delta_median": mean([e["u_over_delta"]["p50"] for e in event]),
        "adam_m_mean": mean([e["adam_m"]["mean"] for e in event]),
    }


def main() -> int:
    runs = sorted((read_run(p) for p in MECH.iterdir() if (p / "mech.jsonl").exists()),
                  key=lambda r: (r["interval"], r["seed"]))
    cost = costs()
    for r in runs:
        r["cost"] = cost.get((r["interval"], r["seed"]))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)

    per: dict[str, dict] = {}
    for n in INTERVALS:
        grp = [r for r in runs if r["interval"] == n]
        paid = [r for r in grp if r["cost"] is not None]
        m = lambda k: sum(r[k] for r in grp) / len(grp)
        per[str(n)] = {
            "runs": len(grp),
            "events": grp[0]["events"],
            "events_per_second_basis": round(grp[0]["events"] / 6280.0, 4),
            "kick_over_drift": round(m("kick_over_drift"), 4),
            "proj_coef": round(m("proj_coef"), 4),
            "u_over_delta_mean": round(m("u_over_delta_mean"), 3),
            "u_over_delta_median": round(m("u_over_delta_median"), 3),
            "cost_seeds": len(paid),
            "cost": round(sum(r["cost"] for r in paid) / len(paid), 6) if paid else None,
        }
    for n in INTERVALS:
        s = per[str(n)]
        if s["cost"] is not None:
            s["cost_per_projection"] = float("%.3e" % (s["cost"] / s["events"]))

    # Two one-parameter readings of the same axis, both fitted on interval 1 only.
    base = per["1"]
    c_freq = base["cost"] / 1.0
    c_erase = base["cost"] / base["kick_over_drift"]
    for n in INTERVALS:
        s = per[str(n)]
        s["pred_frequency"] = round(c_freq / n, 6)
        s["pred_erasure"] = round(c_erase * s["kick_over_drift"] / n, 6)
        if s["cost"] is not None:
            s["ratio_frequency"] = round(s["pred_frequency"] / s["cost"], 2)
            s["ratio_erasure"] = round(s["pred_erasure"] / s["cost"], 2)

    summary = {
        "source": "results_mech/*/mech.jsonl, costs from quant_runs/heldout_eval_v3.csv (heldout_1556, mae_fp32)",
        "definition": "kick_over_drift = 1 - mean(eaten_global) = mean(s/u); proj_coef = mean(proj_global)",
        "per_interval": per,
        "cost_ratio_1_to_20": round(per["1"]["cost"] / per["20"]["cost"], 3),
        "kick_ratio_1_to_20": round(per["20"]["kick_over_drift"] / per["1"]["kick_over_drift"], 2),
        "projection_ratio_1_to_20": round(per["1"]["events"] / per["20"]["events"], 2),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("%-5s %8s %10s %10s %8s %8s %12s" % ("N", "events", "kick/drift", "proj", "u/d", "cost", "cost/proj"))
    for n in INTERVALS:
        s = per[str(n)]
        print("%-5d %8d %10.3f %10.3f %8.2f %8.5f %12.2e"
              % (n, s["events"], s["kick_over_drift"], s["proj_coef"],
                 s["u_over_delta_mean"], s["cost"], s["cost_per_projection"]))
    print("frequency model (c/N):  N=5 %+.5f (x%.2f)  N=20 %+.5f (x%.2f)"
          % (per["5"]["pred_frequency"], per["5"]["ratio_frequency"],
             per["20"]["pred_frequency"], per["20"]["ratio_frequency"]))
    print("erasure model (c*k/N):  N=5 %+.5f (x%.2f)  N=20 %+.5f (x%.2f)"
          % (per["5"]["pred_erasure"], per["5"]["ratio_erasure"],
             per["20"]["pred_erasure"], per["20"]["ratio_erasure"]))
    print("cost ratio 1->20 %.2f, kick ratio %.1f, projection ratio %.1f"
          % (summary["cost_ratio_1_to_20"], summary["kick_ratio_1_to_20"],
             summary["projection_ratio_1_to_20"]))
    print("wrote %s" % OUT_CSV.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
