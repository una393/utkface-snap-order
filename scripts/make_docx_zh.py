# -*- coding: utf-8 -*-
"""把中文 Markdown 稿排成 **A4 双栏期刊版式** 的 Word。只做排版，不改任何文字与数字。
用法: D:/tool/Python/python.exe scripts/make_docx_zh.py [src.md] [out.docx]
"""
import io, os, re, sys
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SONG = "\u5b8b\u4f53"
HEI = "\u9ed1\u4f53"
EN = "Times New Roman"
MATHF = "Cambria Math"
M_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

# ======================= 版式参数（全部集中在此，改版式只改这里） =======================
PAGE_W_CM, PAGE_H_CM = 21.0, 29.7           # A4
MARGIN_TB_CM = 2.4                          # 上下页边距
MARGIN_LR_CM = 1.8                          # 左右页边距
COL_NUM = 2                                 # 正文栏数
COL_SPACE_TWIPS = 454                       # 栏间距（twips，454 ≈ 0.8 cm）
COL_W_CM = 8.3                              # 单栏文本宽
FULL_W_CM = round(PAGE_W_CM - 2 * MARGIN_LR_CM, 2)   # 通栏文本宽 17.4

BODY_PT = 10.0                              # 正文字号
BODY_INDENT_PT = 2 * BODY_PT                # 首行缩进 2 字符 = 20 pt
LINE_SPACING = 1.0                          # 单倍行距
PARA_BEFORE_PT = 0.0                        # 正文档前
PARA_AFTER_PT = 0.0                         # 正文档后

H_SIZES = {1: 11.0, 2: 10.5, 3: 10.0}       # 一/二/三级标题字号
H_BEFORE = {1: 12.0, 2: 6.0, 3: 4.0}        # 段前
H_AFTER = {1: 6.0, 2: 3.0, 3: 2.0}          # 段后
H1_ALIGN_CENTER = True                      # 一级标题居中（沿用旧版对齐，规格未指定）
H_PAGE_BREAK_BEFORE = False                 # 一级标题不另起一页
PAGE_BREAK_TITLES = set()   # 第十六轮起为空集：19 页下强制分页会在 §8 前留 530 pt 空洞，而不分页时 §8 标题与引言同落左栏底部、其后的通栏表另起节，未被栏切        # 这些一级标题强制另起一页（其后紧跟通栏表，避免正文被栏切）

CAPTION_PT = 9.0                            # 图注 / 表题
CAPTION_BEFORE_PT = 4.0
CAPTION_AFTER_PT = 10.0
NOTE_PT = 9.0                               # 表注
TABLE_PT = 8.5                              # 表内文字

TITLE_PT = 18.0                             # 主标题
SUBTITLE_PT = 11.5                          # 副标题
AUTHOR_PT = 10.5                            # 作者行
AFFIL_PT = 9.5                              # 单位行
ABSTRACT_HEAD_PT = 9.0                      # 摘要标题
ABSTRACT_PT = 9.0                           # 摘要正文
KEYWORD_PT = 9.0                            # 关键词行

FRONT_MATTER_FULL_WIDTH = True              # 首部（标题→关键词）单独占通栏节，正文才切双栏
FIG_W_CM = FULL_W_CM                        # 通栏图默认宽
TABLE_FULL_MIN_COLS = 4                     # 列数 ≥ 此值 -> 通栏；否则留在双栏内
TABLE_INCOL_W_CM = COL_W_CM                 # 双栏内表格总宽
TABLE_WEIGHTS = {2: (1.0, 1.0), 3: (1.4, 0.8, 1.0)}   # 每列相对宽；其它列数见 _weights()
TABLE_W_FIRST_BIG = 1.5                     # 列数 ≥4 时首列权重，其余 1.0
TABLE_WEIGHTS_ORD = {3: (2.2, 1.6, 0.7, 1.0, 1.1, 0.9),
                    9: (2.0, 2.2, 2.6, 2.2, 3.0),
                    10: (2.7, 1.5, 1.5, 1.5, 4.6, 2.75, 2.75),
                    13: (2.5, 1.3, 1.3, 1.1, 3.6, 2.6),
                    16: (1.7, 3.6, 3.4, 1.3),
                    19: (1.30, 2.20, 2.20, 1.00)}  # 按表序覆盖列宽权重（表 8 的 CI 列要宽；表 13 是跨后端表、量化代价列要宽；表 15 是决策表、表 18 是路径表）

EQ_PT = 9.5                                 # 显示公式字号
EQ_CENTER_CM = COL_W_CM / 2                 # 公式居中制表位 4.15
EQ_RIGHT_CM = COL_W_CM                      # 公式编号右制表位 8.3
EQ_SPACE_PT = 3.0
EQ_PT_OVERRIDE = {}                         # 逐条缩字号 {公式编号: pt}，栏宽放不下时才填

REF_PT = 9.0                                # 参考文献
REF_HANG_PT = 24.0                          # 悬挂缩进
FOOTER_PT = 9.0                             # 页脚页码
ENABLE_HEADER = False                       # 不要页眉
# =====================================================================================

def set_font(run, size=None, bold=None, east=SONG, color=None, italic=None):
    run.font.name = EN
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    rf.set(qn("w:ascii"), EN); rf.set(qn("w:hAnsi"), EN); rf.set(qn("w:eastAsia"), east)
    if size is not None: run.font.size = Pt(size)
    if bold is not None: run.font.bold = bold
    if italic is not None: run.font.italic = italic
    if color is not None: run.font.color.rgb = color

INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+\*|\x60[^\x60]+\x60|\$[^$]+\$)")

def _math_size(el, size_pt):
    """OMML 里的每个 m:r 显式给字号，否则继承 Normal 样式、无法与正文区分"""
    sz = str(int(round(size_pt * 2)))
    for r in el.iter(M_NS + "r"):
        rPr = r.find(qn("w:rPr"))
        if rPr is None:
            rPr = OxmlElement("w:rPr")
            mpr = r.find(M_NS + "rPr")
            if mpr is not None:
                mpr.addnext(rPr)
            else:
                r.insert(0, rPr)
        rf = rPr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rPr.append(rf)
        rf.set(qn("w:ascii"), MATHF); rf.set(qn("w:hAnsi"), MATHF)
        for tag in ("w:sz", "w:szCs"):
            e = rPr.find(qn(tag))
            if e is None:
                e = OxmlElement(tag); rPr.append(e)
            e.set(qn("w:val"), sz)

def add_math(p, latex, size=None):
    """把一段 LaTeX 转成 Word 原生公式（OMML）并挂到段落上"""
    from mathomml import omml_element
    try:
        el = omml_element(latex.strip())
    except Exception as ex:
        raise SystemExit("公式转换失败: %r (%s)" % (latex, ex))
    if size is not None:
        _math_size(el, size)
    p._p.append(el)
    return p

def add_runs(p, text, size=BODY_PT, east=SONG, base_bold=False):
    """解析 **粗体** / *斜体* / 行内代码 / $行内公式$，其余按普通文本"""
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = p.add_run(tok[2:-2]); set_font(r, size, True, east)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = p.add_run(tok[1:-1]); set_font(r, size, base_bold, east, italic=True)
        elif tok.startswith("\x60") and tok.endswith("\x60"):
            r = p.add_run(tok[1:-1]); set_font(r, size - 0.5, base_bold, east)
            r.font.name = "Consolas"
        elif tok.startswith("$") and tok.endswith("$") and len(tok) > 2:
            add_math(p, tok[1:-1], size)
        else:
            r = p.add_run(tok); set_font(r, size, base_bold, east)

# ---------------------------------------------------------------- 节 / 分栏 / 页脚
def _tune_break_paragraph(doc):
    """add_section 会追加一个只含 sectPr 的空段：压到 1 pt 高、零间距，避免多出一条空行"""
    p = doc.paragraphs[-1]
    pf = p.paragraph_format
    pf.space_before = Pt(0); pf.space_after = Pt(0); pf.line_spacing = LINE_SPACING
    pf.first_line_indent = Pt(0); pf.left_indent = Pt(0)
    pPr = p._p.get_or_add_pPr()
    rPr = pPr.find(qn("w:rPr"))
    if rPr is None:
        rPr = OxmlElement("w:rPr")
        pPr.insert_element_before(rPr, "w:sectPr", "w:pPrChange")
    for tag in ("w:sz", "w:szCs"):
        e = rPr.find(qn(tag))
        if e is None:
            e = OxmlElement(tag); rPr.append(e)
        e.set(qn("w:val"), "2")

def set_cols(sec, num):
    sectPr = sec._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.insert_element_before(cols, "w:formProt", "w:vAlign", "w:noEndnote",
                                     "w:titlePg", "w:textDirection", "w:bidi",
                                     "w:rtlGutter", "w:docGrid", "w:printerSettings",
                                     "w:sectPrChange")
    for ch in list(cols):
        cols.remove(ch)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(COL_SPACE_TWIPS))

def new_section(doc, num):
    """插入连续分节并设定栏数；返回新节（其后的内容落在该节里）"""
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    set_cols(sec, num)
    _tune_break_paragraph(doc)
    return sec

def footer_page_number(section):
    """每节共用的居中 PAGE 域（fldSimple 写法沿用旧版）"""
    section.footer.is_linked_to_previous = False
    p = section.footer.paragraphs[0]
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), " PAGE ")
    r = OxmlElement("w:r"); rPr = OxmlElement("w:rPr")
    rf = OxmlElement("w:rFonts")
    rf.set(qn("w:ascii"), EN); rf.set(qn("w:hAnsi"), EN); rPr.append(rf)
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(FOOTER_PT * 2))); rPr.append(sz)
    r.append(rPr)
    t = OxmlElement("w:t"); t.text = "1"; t.set(qn("xml:space"), "preserve"); r.append(t)
    fld.append(r)
    p._p.append(fld)

# ---------------------------------------------------------------- 段落构件
def body(doc, text, size=BODY_PT, indent=True):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.first_line_indent = Pt(2 * size if indent else 0)
    pf.line_spacing = LINE_SPACING
    pf.space_before = Pt(PARA_BEFORE_PT); pf.space_after = Pt(PARA_AFTER_PT)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_runs(p, text, size)
    return p

def bullet(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Pt(REF_HANG_PT); pf.first_line_indent = Pt(-REF_HANG_PT)
    pf.tab_stops.add_tab_stop(Pt(REF_HANG_PT))
    pf.line_spacing = LINE_SPACING
    pf.space_before = Pt(PARA_BEFORE_PT); pf.space_after = Pt(PARA_AFTER_PT)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_runs(p, "\u2022\t" + text, BODY_PT)
    return p

def centered(doc, text, size, east=SONG, bold=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.first_line_indent = Pt(0); pf.left_indent = Pt(0)
    pf.line_spacing = 1.15
    pf.space_before = Pt(0); pf.space_after = Pt(0)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if bold:
        r = p.add_run(text); set_font(r, size, True, east)
    else:
        add_runs(p, text, size, east)
    return p

def heading(doc, text, level):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(H_BEFORE[level]); pf.space_after = Pt(H_AFTER[level])
    pf.line_spacing = LINE_SPACING
    pf.first_line_indent = Pt(0); pf.left_indent = Pt(0)
    pf.keep_with_next = True
    pf.page_break_before = H_PAGE_BREAK_BEFORE or (level == 1 and text in PAGE_BREAK_TITLES)
    if level == 1:
        p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if H1_ALIGN_CENTER
                       else WD_ALIGN_PARAGRAPH.LEFT)
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_runs(p, text, H_SIZES[level], SONG, base_bold=True)
    for r in p.runs:
        r.font.bold = True
    return p

def caption(doc, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, keep_with_next=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(CAPTION_BEFORE_PT); pf.space_after = Pt(CAPTION_AFTER_PT)
    pf.line_spacing = 1.0
    pf.first_line_indent = Pt(0); pf.left_indent = Pt(0)
    pf.keep_with_next = keep_with_next
    p.alignment = align
    add_runs(p, text, CAPTION_PT)
    return p

def note(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(2); pf.space_after = Pt(8)
    pf.line_spacing = 1.0
    pf.first_line_indent = Pt(0); pf.left_indent = Pt(0)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    m = re.match(r"^(?:\*\*)?\s*(\u6ce8\s*[:\uff1a]\*{0,2}\s*)(.*)$", text.strip())
    if m:
        r = p.add_run(m.group(1).replace("**", "")); set_font(r, NOTE_PT, True, SONG)
        add_runs(p, m.group(2), NOTE_PT)
    else:
        add_runs(p, text, NOTE_PT)
    return p

def picture(doc, path, width_cm):
    if not os.path.exists(path):
        raise SystemExit("图不存在: " + path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(6); pf.space_after = Pt(2)
    pf.first_line_indent = Pt(0); pf.left_indent = Pt(0)
    pf.line_spacing = 1.0
    pf.keep_with_next = True
    p.add_run().add_picture(path, width=Cm(width_cm))
    return p

def spacer(doc, after_pt=6.0):
    sp = doc.add_paragraph()
    pf = sp.paragraph_format
    pf.space_before = Pt(0); pf.space_after = Pt(after_pt)
    pf.line_spacing = LINE_SPACING
    r = sp.add_run(); set_font(r, 6.0)
    return sp

def equation(doc, latex, num):
    size = EQ_PT_OVERRIDE.get(num, EQ_PT)
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(EQ_SPACE_PT); pf.space_after = Pt(EQ_SPACE_PT)
    pf.line_spacing = 1.0
    pf.first_line_indent = Pt(0); pf.left_indent = Pt(0); pf.right_indent = Pt(0)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    ts = pf.tab_stops
    ts.add_tab_stop(Cm(EQ_CENTER_CM), WD_TAB_ALIGNMENT.CENTER)
    ts.add_tab_stop(Cm(EQ_RIGHT_CM), WD_TAB_ALIGNMENT.RIGHT)
    r0 = p.add_run(); set_font(r0, size); r0.add_tab()
    add_math(p, latex, size)
    r1 = p.add_run(); set_font(r1, size); r1.add_tab()
    r2 = p.add_run("(%d)" % num); set_font(r2, size)
    return p

# ---------------------------------------------------------------- 三线表
def _border(el, edge, sz, color="000000"):
    tcPr = el.get_or_add_tcPr() if hasattr(el, "get_or_add_tcPr") else el
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders"); tcPr.append(borders)
    e = OxmlElement("w:" + edge)
    e.set(qn("w:val"), "none" if sz == 0 else "single")
    e.set(qn("w:sz"), str(max(sz, 0)))
    e.set(qn("w:space"), "0"); e.set(qn("w:color"), color)
    borders.append(e)

def _weights(ncol):
    if ncol in TABLE_WEIGHTS:
        return list(TABLE_WEIGHTS[ncol])
    w = [1.0] * ncol
    w[0] = TABLE_W_FIRST_BIG
    return w

def _table_width(t, width_cm, ncol, weights=None):
    tw = int(Cm(width_cm).twips)
    tblPr = t._tbl.tblPr
    el = tblPr.find(qn("w:tblW"))
    if el is None:
        el = OxmlElement("w:tblW")
        tblPr.insert_element_before(el, "w:jc", "w:tblCellSpacing", "w:tblInd",
                                    "w:tblBorders", "w:shd", "w:tblLayout",
                                    "w:tblCellMar", "w:tblLook")
    el.set(qn("w:w"), str(tw)); el.set(qn("w:type"), "dxa")
    ws = list(weights) if weights else _weights(ncol)
    tot = sum(ws)
    widths = [width_cm * x / tot for x in ws]
    for j in range(ncol):
        t.columns[j].width = Cm(widths[j])
    for row in t.rows:
        for j in range(ncol):
            row.cells[j].width = Cm(widths[j])
    return widths

def table(doc, rows, width_cm, weights=None):
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            pf = p.paragraph_format
            pf.space_before = Pt(1); pf.space_after = Pt(1); pf.line_spacing = 1.0
            pf.first_line_indent = Pt(0); pf.left_indent = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            txt = row[j] if j < len(row) else ""
            add_runs(p, txt, TABLE_PT, base_bold=(i == 0))
            _border(cell._tc, "top", 12 if i == 0 else 0)
            _border(cell._tc, "bottom", 12 if i in (0, len(rows) - 1) else 0)
            _border(cell._tc, "left", 0); _border(cell._tc, "right", 0)
    _table_width(t, width_cm, ncol, weights)
    if len(rows) <= 13:
        for ri, row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            if ri < len(rows) - 1:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.keep_with_next = True
    else:
        t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    return t

# ---------------------------------------------------------------- markdown 识别
def split_row(line):
    s = line.strip()
    if s.startswith("|"): s = s[1:]
    if s.endswith("|"): s = s[:-1]
    return [c.strip() for c in s.split("|")]

def is_section_heading(txt):
    t = txt.strip()
    if re.match(r"^[0-9]+\s*[、.．]\s*", t) or re.match(r"^[0-9]+\s+\S", t):
        return True
    return bool(re.match(r"^[\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e]+\s*[\u3001.\uff0e]\s*", t))

def is_fig_caption(txt):
    return bool(re.match(r"^\*{0,2}\s*\u56fe\s*[0-9]+\s*[.\uff0e]", txt.strip()))

def is_tab_caption(txt):
    return bool(re.match(r"^\*{0,2}\s*\u8868\s*[0-9]+\s*[.\uff0e]", txt.strip()))

def is_note(txt):
    return bool(re.match(r"^\*{0,2}\s*\u6ce8\s*[:\uff1a]", txt.strip()))

def next_nonempty(lines, i):
    j = i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    return j

# ---------------------------------------------------------------- 主流程
def has_front_matter(lines):
    """首个正文小节标题之前是否有首部内容（标题/副标题/作者/单位/摘要/关键词）。"""
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("<!--") or s.startswith("---") or s.startswith(">"):
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl, txt = len(m.group(1)), m.group(2).strip()
            if lvl == 1 or "\u6458\u8981" in txt[:4]:
                return True
            return not is_section_heading(txt)
        return True
    return False


def merge_ref_lines(lines):
    """参考文献在 md 里是硬折行的：续行并入上一条，否则每行单独成段、悬挂缩进会错位"""
    start = next((k for k, ln in enumerate(lines)
                  if ln.strip().startswith("#") and "参考文献" in ln), None)
    if start is None:
        return lines
    out, buf, done = list(lines[:start + 1]), None, False
    for ln in lines[start + 1:]:
        s = ln.strip()
        if done:
            out.append(ln); continue
        if s.startswith("#"):
            if buf: out.append(buf); buf = None
            done = True; out.append(ln); continue
        if not s:
            if buf: out.append(buf); buf = None
            out.append(""); continue
        if re.match(r"^\[\d+\]", s):
            if buf: out.append(buf)
            buf = s
        elif buf is not None:
            buf = buf + " " + s
        else:
            out.append(ln)
    if buf: out.append(buf)
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "paper2_snap_order_zh.md"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".docx"
    base = os.path.dirname(os.path.abspath(src))
    lines = merge_ref_lines(io.open(src, encoding="utf-8").read().split("\n"))

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = EN; st.font.size = Pt(BODY_PT)
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    rf.set(qn("w:ascii"), EN); rf.set(qn("w:hAnsi"), EN); rf.set(qn("w:eastAsia"), SONG)
    pf0 = st.paragraph_format
    pf0.space_before = Pt(PARA_BEFORE_PT); pf0.space_after = Pt(PARA_AFTER_PT)
    pf0.line_spacing = LINE_SPACING

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(PAGE_W_CM), Cm(PAGE_H_CM)
    sec.top_margin = sec.bottom_margin = Cm(MARGIN_TB_CM)
    sec.left_margin = sec.right_margin = Cm(MARGIN_LR_CM)
    sec.header_distance = Cm(1.2); sec.footer_distance = Cm(1.2)
    front_full = FRONT_MATTER_FULL_WIDTH and has_front_matter(lines)
    set_cols(sec, 1 if front_full else COL_NUM)
    footer_page_number(sec)

    i, n_tab, n_full_tab, n_img, n_eq, n_note = 0, 0, 0, 0, 0, 0
    refs = False
    full_open = False
    in_front, seen_title, in_abstract = True, False, False
    n_full = 0

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s or s.startswith("<!--") or s.startswith("---") or s.startswith(">"):
            i += 1; continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl, txt = len(m.group(1)), m.group(2).strip()
            if in_front:
                if lvl == 1:
                    centered(doc, txt, TITLE_PT, HEI, bold=True)
                    seen_title = True
                    i += 1; continue
                if "摘要" in txt[:4]:
                    p = heading(doc, txt, 1) if False else None
                    hp = doc.add_paragraph()
                    hp.paragraph_format.space_before = Pt(4)
                    hp.paragraph_format.space_after = Pt(2)
                    hp.paragraph_format.first_line_indent = Pt(0)
                    hp.paragraph_format.keep_with_next = True
                    r = hp.add_run(txt); set_font(r, ABSTRACT_HEAD_PT, True, HEI)
                    in_abstract = True
                    i += 1; continue
                if is_section_heading(txt):
                    in_front = False
                    if front_full:
                        new_section(doc, COL_NUM)
                else:
                    centered(doc, txt, SUBTITLE_PT, SONG)
                    i += 1; continue
            if "参考文献" in txt:
                refs = True
                new_section(doc, 1)          # 书目走通栏：双栏末页会剩下整栏空白
            heading(doc, txt, min(max(lvl - 1, 1), 3))
            i += 1; continue

        if in_front:
            if s.startswith("\u4f5c\u8005\uff1a"):
                centered(doc, s, AUTHOR_PT, SONG); i += 1; continue
            if s.startswith("\u5355\u4f4d\uff1a"):
                centered(doc, s, AFFIL_PT, SONG); i += 1; continue
            if s.startswith("\u5173\u952e\u8bcd\uff1a"):
                p = doc.add_paragraph()
                pf = p.paragraph_format
                pf.first_line_indent = Pt(0); pf.line_spacing = LINE_SPACING
                pf.space_before = Pt(PARA_BEFORE_PT); pf.space_after = Pt(PARA_AFTER_PT)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                lab = "\u5173\u952e\u8bcd\uff1a"
                r = p.add_run(lab); set_font(r, KEYWORD_PT, True, SONG)
                add_runs(p, s[len(lab):], KEYWORD_PT)
                i += 1; continue
            body(doc, s, ABSTRACT_PT if in_abstract else BODY_PT,
                 indent=in_abstract)
            i += 1; continue

        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            n_eq += 1
            equation(doc, s[2:-2], n_eq)
            i += 1; continue

        if s.startswith("!["):
            m2 = re.search(r"\(([^)]+)\)", s)
            wm = re.search(r"!\[[^\]]*\|([0-9.]+)\]", s)
            w_cm = min(float(wm.group(1)), FIG_W_CM) if wm else FIG_W_CM
            new_section(doc, 1)
            picture(doc, os.path.join(base, m2.group(1)), w_cm)
            n_img += 1
            j = next_nonempty(lines, i)
            if j < len(lines) and is_fig_caption(lines[j].strip()):
                caption(doc, lines[j].strip())
                i = j + 1
            else:
                i += 1
            new_section(doc, COL_NUM)
            n_full += 1
            continue

        if s.startswith("* ") or s.startswith("- "):
            bullet(doc, s[2:].strip())
            i += 1; continue

        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = split_row(lines[i])
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in row if c):
                    block.append(row)
                i += 1
            ncol = max(len(r) for r in block)
            wide = ncol >= TABLE_FULL_MIN_COLS
            if wide and not full_open:
                new_section(doc, 1); full_open = True
            table(doc, block, FULL_W_CM if wide else TABLE_INCOL_W_CM,
                  TABLE_WEIGHTS_ORD.get(n_tab + 1))
            n_tab += 1
            if wide:
                j = next_nonempty(lines, i)
                while j < len(lines) and is_note(lines[j].strip()):
                    note(doc, lines[j].strip())
                    n_note += 1
                    i = j + 1
                    j = next_nonempty(lines, i)
                new_section(doc, COL_NUM); full_open = False
                n_full += 1; n_full_tab += 1
            spacer(doc)
            continue

        if is_tab_caption(s):
            j = next_nonempty(lines, i)
            if j < len(lines) and lines[j].strip().startswith("|"):
                if (len(split_row(lines[j])) >= TABLE_FULL_MIN_COLS
                        and not full_open):
                    new_section(doc, 1); full_open = True
                caption(doc, s, WD_ALIGN_PARAGRAPH.LEFT, keep_with_next=True)
                i += 1; continue

        if is_note(s):
            note(doc, s)
            n_note += 1
            i += 1; continue

        if refs:
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Pt(REF_HANG_PT); pf.first_line_indent = Pt(-REF_HANG_PT)
            pf.space_before = Pt(0); pf.space_after = Pt(3); pf.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT   # 书目左对齐：两端对齐遇长 arXiv 号会拉出巨大词距
            add_runs(p, s, REF_PT)
            i += 1; continue

        body(doc, s)
        i += 1

    # 页眉：全部删除
    for hp in doc.element.body.iter(qn("w:sectPr")):
        for ref in hp.findall(qn("w:headerReference")):
            hp.remove(ref)

    doc.save(out)

    b = doc.element.body
    n_omath = len(b.findall(".//" + M_NS + "oMath"))
    n_cantsplit = len(b.findall(".//" + qn("w:cantSplit")))
    sectPrs = list(b.iter(qn("w:sectPr")))
    colv = []
    fref = set()
    for sp in sectPrs:
        c = sp.find(qn("w:cols"))
        colv.append(int(c.get(qn("w:num"))) if c is not None else 1)
        for fr in sp.findall(qn("w:footerReference")):
            fref.add(fr.get(qn("r:id")))
    print("saved %s bytes=%d" % (out, os.path.getsize(out)))
    print("sections=%d  cols=%s" % (len(doc.sections), ",".join(str(x) for x in colv)))
    expect_sec = ((2 if front_full else 1) + 2 * n_full + (1 if refs else 0))
    print("full_width_blocks=%d  front_full=%s  (expect sections = %d)" % (n_full, front_full, expect_sec))
    print("oMath=%d  cantSplit=%d  inline_shapes=%d  tables=%d  paragraphs=%d"
          % (n_omath, n_cantsplit, len(doc.inline_shapes), len(doc.tables), len(doc.paragraphs)))
    print("footer_parts=%d  header_refs=%d  equations=%d  images=%d  tables_md=%d (wide %d)"
          % (len(fref), len(b.findall(".//" + qn("w:headerReference"))), n_eq, n_img, n_tab, n_full_tab))
    print("eq_pt_override=%s  eq_pt=%.2f" % (EQ_PT_OVERRIDE, EQ_PT))
    print("notes=%d" % n_note)

if __name__ == "__main__":
    main()
