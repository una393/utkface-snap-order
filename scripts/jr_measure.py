# -*- coding: utf-8 -*-
"""探针 PDF 读数：页数 / 图 bbox / 公式编号 x / 残留标记 / 页码 / 越界"""
import io, re, sys
import fitz

CM = 72.0 / 2.54
PW, PH = 595.28, 841.89
L_EDGE = 1.8 * CM                      # 51.02
R_EDGE = PW - 1.8 * CM                 # 544.26
COLW = 8.3 * CM                        # 235.28
GUT = 0.8 * CM
COL_R_LEFT = L_EDGE + COLW             # 左栏右缘 286.31
COL_R_RIGHT = PW - 1.8 * CM            # 右栏右缘
ROWMID = L_EDGE + COLW + GUT           # 右栏左缘 309.0

pdf = sys.argv[1]
out = []
def p(*a): out.append(" ".join(str(x) for x in a))

doc = fitz.open(pdf)
p("pages", doc.page_count)

# 图
imgs = []
for pno in range(doc.page_count):
    for inf in doc[pno].get_image_info():
        b = inf["bbox"]
        imgs.append((pno + 1, b[0], b[1], b[2], b[3]))
p("=== images (%d) ===" % len(imgs))
for k, (pg, x0, y0, x1, y1) in enumerate(imgs, 1):
    p("img#%d page=%-2d x0=%7.2f x1=%7.2f w=%7.2fpt (%5.2fcm)  Ldev=%6.2f Rdev=%6.2f  y=%7.2f..%7.2f"
      % (k, pg, x0, x1, x1 - x0, (x1 - x0) / CM, x0 - L_EDGE, R_EDGE - x1, y0, y1))

# 公式编号
p("=== equation numbers ===")
nums = []
for pno in range(doc.page_count):
    page = doc[pno]
    words = page.get_text("words")
    for w in words:
        if re.fullmatch(r"\(\d{1,2}\)", w[4]):
            nums.append((int(w[4][1:-1]), pno + 1, w[0], w[2], w[1], w[5], w[6]))
nums.sort()
for (n, pg, x0, x1, y, blk, lin) in nums:
    col = "L" if x1 < ROWMID else "R"
    edge = COL_R_LEFT if col == "L" else COL_R_RIGHT
    p("eq(%2d) page=%-2d col=%s x0=%7.2f x1=%7.2f  dev_from_col_edge=%7.2f"
      % (n, pg, col, x0, x1, x1 - edge))
xs = [x[3] for x in nums]
if xs:
    p("num_count=%d  x1 min=%.2f max=%.2f spread=%.2f" % (len(xs), min(xs), max(xs), max(xs) - min(xs)))

# ==== 公式行几何：按 y 带聚行，量数学区右端与编号左端的关系 ====
p("=== equation rows: math extent vs number ===")
rows_all = []
for pno in range(doc.page_count):
    page = doc[pno]
    ws = [w for w in page.get_text("words")]
    ws.sort(key=lambda z: (round((z[1] + z[3]) / 2.0, 1), z[0]))
    bands = []
    for w in ws:
        yc = (w[1] + w[3]) / 2.0
        if bands and abs(bands[-1][0] - yc) <= 4.0:
            bands[-1][1].append(w)
        else:
            bands.append([yc, [w]])
    for yc, bw in bands:
        if not bw: continue
        bw.sort(key=lambda z: z[0])
        last = bw[-1]
        if not re.fullmatch(r"\(\d{1,2}\)", last[4]): continue
        left = [w for w in bw if w[2] <= last[0] + 0.5]
        if not left: continue
        rows_all.append((int(last[4][1:-1]), pno + 1, min(w[0] for w in left),
                         max(w[2] for w in left), last[0], last[2], yc))
for (n, pg, mx0, mx1, nx0, nx1, yc) in rows_all:
    col_edge = COL_R_LEFT if nx1 < ROWMID else COL_R_RIGHT
    col_left = L_EDGE if nx1 < ROWMID else ROWMID
    p("eq(%2d) p%-2d y=%6.1f  math=[%7.2f,%7.2f] w=%6.2fpt(%5.2fcm)  numx0=%7.2f  gap=%6.2f  numx1-edge=%6.2f  avail=%6.2f"
      % (n, pg, yc, mx0, mx1, mx1 - mx0, (mx1 - mx0) / CM, nx0, nx0 - mx1, nx1 - col_edge,
         col_edge - col_left - (nx1 - nx0)))
bad_overlap = [r for r in rows_all if r[4] - r[3] < 0.5]
p("rows_with_number=%d  overlap_or_touch=%d" % (len(rows_all), len(bad_overlap)))
for r in bad_overlap:
    p("   OVERLAP eq", r[0], "page", r[1], "mathx1=%.2f numx0=%.2f" % (r[3], r[4]))

# 图与题注同页
p("=== figure caption pages ===")
cap = {}
for pno in range(doc.page_count):
    for line in doc[pno].get_text().split("\n"):
        m = re.match(r"^\s*\u56fe\s*(\d+)\s*\.", line)
        if m:
            cap.setdefault(int(m.group(1)), pno + 1)
for k, (pg, x0, y0, x1, y1) in enumerate(imgs, 1):
    p("fig%d image_page=%d caption_page=%s same=%s"
      % (k, pg, cap.get(k), cap.get(k) == pg))

# 残留标记 / 越界 / 页码
txt = "".join(doc[i].get_text() for i in range(doc.page_count))
p("=== residue ===")
p("star_pair_count=%d  dollar_count=%d  backtick=%d" % (txt.count("**"), txt.count("$"), txt.count("`")))
over = []
for pno in range(doc.page_count):
    for w in doc[pno].get_text("words"):
        if w[2] > R_EDGE + 1.0 or w[0] < L_EDGE - 1.0:
            over.append((pno + 1, round(w[0], 1), round(w[2], 1), w[4][:14]))
p("out_of_margin_words=%d" % len(over))
for o in over[:12]:
    p("   overflow", o)

p("=== footer page numbers ===")
bad = []
for pno in range(doc.page_count):
    page = doc[pno]
    cand = [w for w in page.get_text("words")
            if re.fullmatch(r"\d{1,3}", w[4]) and w[1] > PH - 2.4 * CM - 6]
    if not cand:
        bad.append((pno + 1, "none")); continue
    w = cand[0]
    c = (w[0] + w[2]) / 2.0
    if abs(c - PW / 2.0) > 6.0 or w[4].strip() != str(pno + 1):
        bad.append((pno + 1, "off center=%.1f val=%s" % (c, w[4])))
p("footer_bad=%d %s" % (len(bad), bad[:8]))

p("=== page fill (last text y per page) ===")
row = []
for pno in range(doc.page_count):
    ys = [w[3] for w in doc[pno].get_text("words") if w[3] < PH - 1.8 * CM - 4]
    row.append("%d:%.0f" % (pno + 1, max(ys) if ys else 0))
p(" ".join(row))

io.open("drafts/preview/jr_measure.txt", "w", encoding="utf-8").write("\n".join(out))
print("ok", len(out))
p("=== all equation numbers near column edge ===")
cands = []
for pno in range(doc.page_count):
    for w in doc[pno].get_text("words"):
        if not re.fullmatch(r"\(\d{1,2}\)", w[4]): continue
        d1 = abs(w[2] - COL_R_RIGHT); d2 = abs(w[2] - COL_R_LEFT)
        if min(d1, d2) <= 1.5:
            cands.append((int(w[4][1:-1]), pno + 1, w[0], w[2], "R" if d1 < d2 else "L"))
cands.sort()
for c in cands:
    p("num(%2d) page=%d col=%s x0=%7.2f x1=%7.2f edge_dev=%+6.2f" % (c[0], c[1], c[4], c[2], c[3], c[3] - (COL_R_RIGHT if c[4]=="R" else COL_R_LEFT)))
p("count=%d unique=%d numbers=%s" % (len(cands), len({c[0] for c in cands}), sorted({c[0] for c in cands})))
