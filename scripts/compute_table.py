"""Per-family device time, reconstructed from the per-epoch `sec` column that
train.py writes into every run's history.csv.

The manuscript used to report only run counts for the families outside the five
main sequences, because result.json carries no wall-clock field. history.csv does:
one row per epoch with the seconds that epoch took (training plus the monitoring
evaluation). Summing the column per run gives the training time actually spent,
without re-running anything and without touching the GPU.

Writes quant_runs/compute_table.csv (one row per run) and prints the family
summary that the manuscript quotes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

FIELDS = ["family", "run", "dataset", "arch", "head", "ema_mode", "snap_every",
          "quant_mode", "lr", "epochs", "epochs_logged", "sec_total", "sec_per_epoch"]


def family_of(d: Path, a: dict) -> str:
    ds = a.get("dataset", "utkface")
    ep = a.get("epochs")
    lr = float(a.get("lr", 0) or 0)
    head = a.get("head", "relu_qat")
    name = d.parts[0]
    if name == "results_v1_buggy":
        return "withdrawn (deployment-table bug)"
    if name == "results_sr":
        return "stochastic-rounding control"
    if name == "results_mech":
        return "mechanism runs"
    if ds == "cifar10":
        return "CIFAR-10 arm (60 ep)"
    if head != "relu_qat":
        return "head controls (GELU / GLU)"
    if name in ("results_ext_lr1e4", "results_ext_mid20"):
        return "architecture axis (ResNet-18, lr 1e-4)"
    if ds == "koniq":
        return "dataset axis (KonIQ-10k)"
    if ep and ep != 20:
        return "epoch control (60 ep)"
    if lr >= 2.5e-3:
        return "learning-rate control (lr 3e-3)"
    if lr >= 5e-4:
        return "learning-rate control (lr 1e-3)"
    return "main sequences (UTKFace, 20 ep, lr 1e-4)"


def main() -> None:
    rows = []
    skipped = []
    for rj in sorted(Path(".").glob("results*/**/result.json")):
        d = rj.parent
        a = json.loads(rj.read_text(encoding="utf-8")).get("args", {})
        hist = d / "history.csv"
        if not hist.exists():
            skipped.append(str(d))
            continue
        recs = list(csv.DictReader(hist.open(encoding="utf-8")))
        try:
            secs = [float(r["sec"] or 0) for r in recs]
        except KeyError:
            skipped.append(str(d))
            continue
        total = sum(secs)
        if total <= 0:
            skipped.append(str(d))
            continue
        rows.append({
            "family": family_of(d, a),
            "run": d.name,
            "dataset": a.get("dataset", "utkface"),
            "arch": a.get("arch", "regnety"),
            "head": a.get("head", "relu_qat"),
            "ema_mode": a.get("ema_mode", "none"),
            "snap_every": a.get("snap_every", 0),
            "quant_mode": a.get("quant_mode", "overwrite"),
            "lr": a.get("lr"),
            "epochs": a.get("epochs"),
            "epochs_logged": len(recs),
            "sec_total": round(total, 1),
            "sec_per_epoch": round(total / max(1, len(recs)), 1),
        })

    out = Path("quant_runs/compute_table.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    tot = sum(r["sec_total"] for r in rows)
    fam = {}
    for r in rows:
        s = fam.setdefault(r["family"], [0, 0.0, 0.0])
        s[0] += 1
        s[1] += r["sec_total"]
        s[2] += r["sec_per_epoch"]
    print("runs with logged time: %d   skipped: %d" % (len(rows), len(skipped)))
    print("total logged training time: %.2f device hours" % (tot / 3600.0))
    print()
    print("%-58s %5s %9s %9s" % ("family", "runs", "hours", "s/run"))
    for k in sorted(fam, key=lambda x: -fam[x][1]):
        n, s, _ = fam[k]
        print("%-58s %5d %9.2f %9.0f" % (k, n, s / 3600.0, s / n))
    print()
    print("wrote", out)


if __name__ == "__main__":
    main()
