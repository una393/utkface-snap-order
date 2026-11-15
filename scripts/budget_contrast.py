"""Deployment-MAE contrast for the budget (60 epoch) and learning-rate (3e-3) axes.

Reuses round20_mechanism.py's loader/evaluator and its mechanism_cache, so only the
new run directories are scored. Prints the held-out 1,556 contrast for each axis.
"""
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import round20_mechanism as rm

CACHE = Path("quant_runs/mechanism_cache")
OUT = Path("quant_runs/budget_contrast.json")


def cache_path(dirspec: str) -> Path:
    return CACHE / (dirspec.replace("/", "_").replace(chr(92), "_") + ".json")


def main() -> None:
    a = __import__("argparse").ArgumentParser()
    a.add_argument("--raw", default=rm.RAW)
    a.add_argument("--data", default=rm.DATA)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--threads", type=int, default=8)
    a.add_argument("--bs", type=int, default=128)
    a = a.parse_args()

    long_ = rm.select_dir_group(Path("results_long"), "long60", 1e-4, 60, (0, 1), tuple(range(8)))
    lr3 = [e for e in rm.select_lr() if e["group"] == "lr3e-3"]
    main_ = rm.select_main()
    ents = long_ + lr3 + main_
    print("runs: long60=%d lr3e-3=%d main20=%d" % (len(long_), len(lr3), len(main_)), flush=True)
    for g, es in (("long60", long_), ("lr3e-3", lr3)):
        seen = sorted({(e["N"], e["seed"]) for e in es})
        print("  %-7s cells=%d seeds=%s" % (g, len(seen),
              sorted({s for _, s in seen})), flush=True)

    tasks = [{"run": e["run"], "dir": e["dir"], "sp": e["sp"], "ema_mode": e["ema_mode"]}
             for e in ents]
    R = {}
    pending = []
    for t in tasks:
        cp = cache_path(t["dir"])
        if cp.exists():
            try:
                R[t["run"]] = json.loads(cp.read_text(encoding="utf-8"))
                continue
            except Exception:
                pass
        pending.append(t)
    print("cache hit %d, to score %d" % (len(R), len(pending)), flush=True)

    if pending:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"),
                                 initializer=rm._init_worker,
                                 initargs=(a.raw, a.data, a.bs, a.threads)) as ex:
            for k, r in enumerate(ex.map(rm.eval_task, pending), 1):
                R[r["run"]] = r
                cache_path(pending[k - 1]["dir"]).write_text(
                    json.dumps(r, ensure_ascii=False), encoding="utf-8")
                print("  [%2d/%d] %-46s %5.1fs" % (k, len(pending), r["run"], r["seconds"]),
                      flush=True)
        print("scoring %.1f s" % (time.time() - t0), flush=True)

    def cells(ents_g):
        out = {}
        for n in (1, 20, 0):
            es = [e for e in ents_g if e["N"] == n]
            if es:
                out[n] = rm.cell_of(es, R, "heldout_1556")
        return out

    rep = {"groups": {}, "contrasts": {}}
    for name, ents_g, pairs in (("long60", long_, ((1, 0), (1, 20))),
                                ("lr3e-3", lr3, ((1, 0), (20, 0))),
                                ("main20", main_, ((1, 0),))):
        c = cells(ents_g)
        rep["groups"][name] = {str(n): {"n": v["n"], "seeds": v["seeds"],
                                        "fp32": v["fp32"], "b8": v["b8"]} for n, v in c.items()}
        for hi, lo in pairs:
            if hi in c and lo in c:
                key = "%s:N%d_minus_N%d" % (name, hi, lo)
                con = rm.contrast_of([e for e in ents_g if e["N"] == hi],
                                     [e for e in ents_g if e["N"] == lo],
                                     R, "heldout_1556", key)
                rep["contrasts"][key] = con

    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for key, con in rep["contrasts"].items():
        for prec in ("fp32", "b8"):
            s = con[prec]
            print("%-26s %-4s mean %+0.6f  CI [%+0.6f, %+0.6f]  p=%s  n=%d  seeds=%s"
                  % (key, prec, s["mean"], s["ci_lo"], s["ci_hi"],
                     "%.3g" % s["p"] if s["p"] is not None else "-", s["n"], con["seeds"]),
                  flush=True)
        print("   per-seed fp32: %s" % ", ".join("%+.5f" % d for d in con["fp32"]["diffs"]),
              flush=True)
    print("wrote %s" % OUT, flush=True)


if __name__ == "__main__":
    sys.exit(main())
