
import os, subprocess, sys
from win32com.client import Dispatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD   = os.path.join(ROOT, "drafts", "paper2_snap_order_zh.md")
DOCX = os.path.join(ROOT, "drafts", "paper2_snap_order_zh.docx")
PDF  = os.path.join(ROOT, "drafts", "paper2_snap_order_zh.pdf")
PY   = r"D:\tool\Python\python.exe"

EXPECT_PAGES, EXPECT_WORDS = 30, 34267  # 由 build 后实测回填

r = subprocess.run([PY, os.path.join(ROOT, "scripts", "make_docx_zh.py"), MD, DOCX],
                   capture_output=True, text=True, encoding="utf8")
print("BUILD:", r.stdout.strip()[-400:], "| rc", r.returncode)
if r.returncode != 0:
    print(r.stderr[-800:]); sys.exit(1)
print("docx bytes", os.path.getsize(DOCX))

pages = words = -1
app = Dispatch("Word.Application"); app.Visible = False
try:
    d = app.Documents.Open(DOCX, ReadOnly=False)
    d.Fields.Update()
    d.SaveAs2(PDF, FileFormat=17)
    pages, words = d.ComputeStatistics(2), d.ComputeStatistics(0)
    print("pages:", pages, "words:", words)
    d.Close(False)
finally:
    app.Quit()
if (pages, words) != (EXPECT_PAGES, EXPECT_WORDS):
    print("MISMATCH pages/words = %s/%s, 记录值 %s/%s" % (pages, words, EXPECT_PAGES, EXPECT_WORDS))
    sys.exit(1)
print("pdf bytes", os.path.getsize(PDF))
