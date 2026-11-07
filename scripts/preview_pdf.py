
import os, pymupdf
doc = pymupdf.open(r"C:\Users\24399\Desktop\paper-utkface\drafts\paper2_snap_order_zh.pdf")
print("pages", doc.page_count)
want = {"3.2": None, "5.1": None, "3.4": None, "3.3": None}
for i, p in enumerate(doc):
    t = p.get_text()
    for k in want:
        if want[k] is None and k in t:
            want[k] = i
print(want)
out = r"C:\Users\24399\Desktop\paper-utkface\drafts\preview"
for k, i in want.items():
    if i is None: continue
    doc[i].get_pixmap(dpi=110).save(os.path.join(out, "eq_%s.png" % k.replace(".", "")))
print("ok")
