
import os
from win32com.client import Dispatch
src = r"C:/Users/USER\Desktop\paper-utkface\drafts\paper2_snap_order_zh.docx"
dst = r"C:/Users/USER\Desktop\paper-utkface\drafts\paper2_snap_order_zh.pdf"
app = Dispatch("Word.Application"); app.Visible = False
try:
    d = app.Documents.Open(src, ReadOnly=False)
    d.Fields.Update()
    d.SaveAs2(dst, FileFormat=17)
    print("pages:", d.ComputeStatistics(2))
    d.Close(False)
finally:
    app.Quit()
print("pdf bytes", os.path.getsize(dst))
