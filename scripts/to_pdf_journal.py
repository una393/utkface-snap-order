
import os
from win32com.client import Dispatch
src = r"C:\Users\24399\Desktop\paper-utkface\drafts\_journal_probe.docx"
dst = r"C:\Users\24399\Desktop\paper-utkface\drafts\_journal_probe.pdf"
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
