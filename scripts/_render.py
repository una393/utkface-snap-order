
import os, pymupdf
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pymupdf.open(os.path.join(_ROOT, "drafts", "paper2_snap_order_zh.pdf"))
print("pages", len(d))
H = d[0].rect.height
for i, p in enumerate(d):
    ys = [b[3] for b in p.get_text("blocks") if b[4].strip()] + [im["bbox"][3] for im in p.get_image_info()]
    gap = 1 - (max(ys) if ys else 0) / H
    if gap > 0.15:
        print("page", i + 1, "gap %.0f%%" % (gap * 100))
for i in (4, 7, 8, 9):
    d[i].get_pixmap(dpi=105).save(r"C:/Users/USER\Desktop\paper-utkface\drafts\preview\pp%d.png" % (i + 1))
print("done")
