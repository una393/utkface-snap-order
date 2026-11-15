# -*- coding: utf-8 -*-
import io, re, sys
src = sys.argv[1]
lines = io.open(src, encoding="utf-8").read().split("\n")
out = []
def p(*a):
    out.append(" ".join(str(x) for x in a))
N = len(lines)
def prev_nonempty(i):
    j = i - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    return j
def next_nonempty(i):
    j = i + 1
    while j < N and not lines[j].strip():
        j += 1
    return j
p("total lines", N)
p("=== images: prev / self / next ===")
for i, ln in enumerate(lines):
    if ln.strip().startswith("!["):
        a = prev_nonempty(i); b = next_nonempty(i)
        p("L%-4d prev[%d]=%s" % (i + 1, a + 1, lines[a].strip()[:46]))
        p("        self       =%s" % ln.strip()[:46])
        p("        next[%d]=%s" % (b + 1, lines[b].strip()[:46]))
p("=== tables: prev / headrow / tail ===")
i = 0
while i < N:
    if lines[i].strip().startswith("|"):
        j = i
        while j < N and lines[j].strip().startswith("|"):
            j += 1
        ncol = max(len([c for c in lines[k].strip().strip("|").split("|")]) for k in range(i, j))
        a = prev_nonempty(i); b = next_nonempty(j - 1)
        p("L%-4d rows=%-3d cols=%-2d prev[%d]=%s" % (i + 1, j - i, ncol, a + 1, lines[a].strip()[:40]))
        p("        after[%d]=%s" % (b + 1, lines[b].strip()[:40]))
        i = j
    else:
        i += 1
p("=== misc ===")
p("display", sum(1 for ln in lines if ln.strip().startswith("$$")), "bullets", sum(1 for ln in lines if ln.strip().startswith("* ") or ln.strip().startswith("- ")))
p("ordered", sum(1 for ln in lines if re.match(r"^\d+\.\s", ln.strip())))
p("quote", sum(1 for ln in lines if ln.strip().startswith(">")))
p("comment", sum(1 for ln in lines if ln.strip().startswith("<!--")))
p("hr", sum(1 for ln in lines if ln.strip() == "---"))
io.open("drafts/preview/jr_struct.txt", "w", encoding="utf-8").write("\n".join(out))
print("ok", len(out))
