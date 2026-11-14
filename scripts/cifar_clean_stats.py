
import csv, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import topup_table as tt

STATE = ROOT / "logs" / "cifar_clean_runs.jsonl"
OUT = ROOT / "quant_runs" / "cifar_clean_split.csv"
rows = [json.loads(l) for l in STATE.read_text(encoding="utf-8").splitlines() if l.strip()]
rec = {}
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    w = csv.writer(f)
    w.writerow(["arm", "seed", "run", "val_err_live", "test_err_live"])
    for r in rows:
        run = Path(r["dir"]).name
        j = json.loads((ROOT / r["dir"] / "result.json").read_text(encoding="utf-8"))
        rec[(r["arm"], int(r["seed"]))] = j
        w.writerow([r["arm"], r["seed"], run,
                    "%.6f" % j["val_err_live"], "%.6f" % j["test_err_live"]])

d = [rec[("live@1", s)]["test_err_live"] - rec[("live@inf", s)]["test_err_live"] for s in range(8)]
n = len(d); m = sum(d) / n
sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)); se = sd / math.sqrt(n)
t = m / se
p = tt._betai((n - 1) / 2.0, 0.5, (n - 1) / ((n - 1) + t * t))
tc = 2.3646
off = sum(rec[("live@inf", s)]["test_err_live"] for s in range(8)) / 8
one = sum(rec[("live@1", s)]["test_err_live"] for s in range(8)) / 8
print("projection off %.4f (%.2f %%)" % (off, 100 * (1 - off)))
print("interval 1     %.4f (%.2f %%)" % (one, 100 * (1 - one)))
print("A %+.3f pp  sd %.3f  se %.3f  t %.2f  p %.2e  CI [%+.3f, %+.3f]  %d/8"
      % (100 * m, 100 * sd, 100 * se, t, p, 100 * (m - tc * se), 100 * (m + tc * se),
         sum(1 for x in d if x > 0)))
print("wrote", OUT.relative_to(ROOT))
