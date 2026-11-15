"""ext_deploy.py 的 csv -> 论文可粘贴的两轴 Markdown 表（纯 csv、纯计算，无 torch/GPU/npz）。
Δ 约定同稿子 §3.4：Δ = 被检验项 − 参照项，正值 = 更差；端点对照取 Δ = MAE(N=1) − MAE(N=0)，
即「每步就地覆写」相对「完全不投影」的代价。N = 0（train.py 的 --snap-every 0）显示为 N = ∞。
用法：python scripts/ext_tables.py --csv a.csv[,b.csv] --label arch|data [--out out.md]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path
from statistics import mean

try:
    sys.stdout.reconfigure(encoding="utf-8")  # 控制台默认 GBK：Δ 的 U+2212 会让 write 崩
    _U8 = True
except (AttributeError, ValueError):  # stdout 不可重配 → 退化为 ASCII 负号
    _U8 = False

COLS = ["run", "dataset", "subset", "arch", "head", "mode", "N", "split_seed", "seed", "n", "mae_p32", "mae_8", "mae_6", "mae_4", "source"]
HEAD = ["seed", "协议", "间隔", "图数 n", "FP32", "8 bit", "6 bit", "4 bit"]
BITS = ["mae_p32", "mae_8", "mae_6", "mae_4"]
MODE_ORD = {"none": 0, "before_snap": 1, "after_snap": 2}
LABEL = {"arch": "架构轴", "data": "数据轴"}
def f6(x): return "%.6f" % x
def s6(x): return "%+.6f" % x
def arc(n): return "N = ∞" if n == 0 else "N = %d" % n
def mkey(m): return (MODE_ORD.get(m, len(MODE_ORD)), m)
def nkey(n): return (1 if n == 0 else 0, n)
def rule(k): return (k[0],) + mkey(k[1]) + nkey(k[2])
def ncell(rs): ns = sorted({int(r["n"]) for r in rs}); return str(ns[0]) if len(ns) == 1 else "—"

def md(head, body): return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in body]

def load(paths):
    rows = []
    for p in paths:
        with Path(p).open(newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            miss = [c for c in COLS if c not in (rd.fieldnames or [])]
            if miss:
                raise ValueError("%s: 缺列 %s；现有列 %s" % (p, miss, rd.fieldnames))
            rows += [dict(r, _csv=str(p)) for r in rd]
    if not rows:
        raise ValueError("没有数据行：%s" % ", ".join(paths))
    return rows

def check(ds, rows):
    """(seed, mode, N) -> row。重复键 raise；行数与 n 异常只 WARNING（以后可能加 seed）。"""
    g = OrderedDict()
    for r in rows:
        k = (int(r["seed"]), r["mode"], int(r["N"]))
        if k in g:
            raise ValueError("dataset=%s: 重复行 (seed=%d, mode=%s, N=%d)，%s / %s" % (ds, k[0], k[1], k[2], g[k]["_csv"], r["_csv"]))
        g[k] = r
    if len(rows) != 8:
        print("WARNING expected 8 rows, got %d (dataset=%s)" % (len(rows), ds), file=sys.stderr)
    ns = sorted({int(r["n"]) for r in rows})
    if len(ns) > 1:
        print("WARNING n varies: %s (dataset=%s)" % (ns, ds), file=sys.stderr)
    return g

def inner_table(g):
    """行 = (seed, mode, N)：seed 升序、none<before_snap<after_snap、N=1 先 N=∞ 后；末尾两 seed 均值行。"""
    body = [[str(k[0]), k[1], arc(k[2]), ncell([g[k]])] + [f6(float(g[k][c])) for c in BITS] for k in sorted(g, key=rule)]
    cells = OrderedDict()
    for k in sorted(g, key=rule):
        cells.setdefault((k[1], k[2]), []).append(g[k])
    for (mode, n), rs in sorted(cells.items(), key=lambda kv: mkey(kv[0][0]) + nkey(kv[0][1])):
        body.append(["两 seed 均值", mode, arc(n), ncell(rs)] + [f6(mean(float(r[c]) for r in rs)) for c in BITS])
    return md(HEAD, body)

def endpoints(g):
    grid = OrderedDict()
    for (seed, mode, n), r in g.items():
        grid.setdefault(mode, OrderedDict()).setdefault(seed, {})[n] = r
    out = []
    for mode in sorted(grid, key=mkey):
        ds, per = [], grid[mode]
        for seed in sorted(per):
            cell = per[seed]
            if 1 not in cell or 0 not in cell:
                raise ValueError("mode=%s seed=%d: 端点缺行，需要 N=1 与 N=0，实有 %s" % (mode, seed, sorted(cell)))
            ds.append([float(cell[1][c]) - float(cell[0][c]) for c in BITS])
        out.append((mode, list(zip(sorted(per), ds)), [mean(d[i] for d in ds) for i in range(len(BITS))]))
    return out

def render(label, ds, rows):
    g = check(ds, rows)
    pick = lambda c: "/".join(sorted({r[c] for r in rows}))
    out = ["## %s · dataset=%s" % (LABEL.get(label, label), ds), "",
           "`subset=%s` / `arch=%s` / `head=%s` / %d 行 / n = %s"
           % (pick("subset"), pick("arch"), pick("head"), len(rows), "/".join(str(int(x)) for x in sorted({int(r["n"]) for r in rows}))), "",
           "Δ 约定（§3.4）：Δ = 被检验项 − 参照项，正值 = 更差；端点对照取 Δ = MAE(N=1) − MAE(N=∞)，正值表示「每步就地覆写」更差。", "",
           "### 轴内表", ""] + inner_table(g) + ["", "### 端点对照（Δ = MAE(N=1) − MAE(N=∞)）", ""]
    body = []
    for mode, ds_, avg in endpoints(g):
        body += [[mode, str(s), s6(d[0]), s6(d[1])] for s, d in ds_]
        body.append([mode, "两 seed 均值", s6(avg[0]), s6(avg[1])])
    one = [r for k, r in g.items() if k[2] == 1]
    zero = [r for k, r in g.items() if k[2] == 0]
    if not one or not zero:
        raise ValueError("dataset=%s: 缺 N=1 或 N=0 的行，无法算端点摘要" % ds)
    segs = []
    for i in range(2):
        m1, m0 = mean(float(r[BITS[i]]) for r in one), mean(float(r[BITS[i]]) for r in zero)
        segs.append("%s: N=1 均值 %s / N=∞ 均值 %s / Δ %s" % (HEAD[4 + i], f6(m1), f6(m0), s6(m1 - m0)))
    out += md(["协议", "seed", "Δ FP32", "Δ 8 bit"], body)
    out += ["", "### 一行摘要", "",
            "（块内 N=1 共 %d 行、N=∞ 共 %d 行，对全部 mode 与 seed 求平均）" % (len(one), len(zero)), "", "；".join(segs), ""]
    return out

def main():
    p = argparse.ArgumentParser(description="ext_deploy csv -> 两轴 Markdown 表（纯 csv 计算）")
    p.add_argument("--csv", required=True, help="csv 路径，逗号分隔多个")
    p.add_argument("--label", choices=["arch", "data"], required=True)
    p.add_argument("--out", default="", help="输出 md 路径；缺省打 stdout")
    a = p.parse_args()
    paths = [s.strip() for s in a.csv.split(",") if s.strip()]
    try:
        blocks = OrderedDict()
        for r in load(paths):
            blocks.setdefault(r["dataset"], []).append(r)
        text = "\n".join("\n".join(render(a.label, d, rs)) for d, rs in blocks.items()) + "\n"
    except ValueError as e:
        raise SystemExit("ERROR: %s" % e)
    if not a.out:
        sys.stdout.write(text if _U8 else text.replace("\u2212", "-"))
        return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print("wrote %s (%d blocks, %d chars)" % (a.out, len(blocks), len(text)))
if __name__ == "__main__":
    main()
