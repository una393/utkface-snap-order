# -*- coding: utf-8 -*-
"""Markdown (this repo's subset) -> Springer Nature LaTeX (sn-jnl).

Usage:
  python scripts/make_latex.py --md drafts/paper2_snap_order.md --out drafts/submission --name main
  python scripts/make_latex.py --md drafts/paper2_esm_en.md --out drafts/submission --name esm --esm
"""
from __future__ import annotations
import argparse, os, re, shutil

BT = chr(96)
BS = chr(92)
EXTRA = {"\u221e": r"$\infty$", "\u2218": r"$\circ$", "\u00b5": r"$\mu$",
         "\u00b2": r"$^2$", "\u00f7": r"$\div$", "\u00e1": r"\'a"}
UNI = {
    "\u2212": "-", "\u2013": "--", "\u2014": "---", "\u00b1": r"$\pm$", "\u00d7": r"$\times$",
    "\u2264": r"$\le$", "\u2265": r"$\ge$", "\u2260": r"$\neq$", "\u2248": r"$\approx$",
    "\u2192": r"$\to$", "\u2190": r"$\leftarrow$", "\u00b7": r"$\cdot$", "\u00b0": r"$^\circ$",
    "\u03b1": r"$\alpha$", "\u03b2": r"$\beta$", "\u03b3": r"$\gamma$", "\u03b4": r"$\delta$",
    "\u03b5": r"$\epsilon$", "\u03b7": r"$\eta$", "\u03b8": r"$\theta$", "\u03ba": r"$\kappa$",
    "\u03bb": r"$\lambda$", "\u03bc": r"$\mu$", "\u03bd": r"$\nu$", "\u03c0": r"$\pi$",
    "\u03c1": r"$\rho$", "\u03c3": r"$\sigma$", "\u03c4": r"$\tau$", "\u03c6": r"$\phi$",
    "\u03c7": r"$\chi$", "\u03c8": r"$\psi$", "\u03c9": r"$\omega$", "\u0393": r"$\Gamma$",
    "\u0394": r"$\Delta$", "\u03a3": r"$\Sigma$", "\u03a6": r"$\Phi$",
    "\u201c": BT + BT, "\u201d": "''", "\u2018": BT, "\u2019": "'", "\u2026": r"\ldots{}",
    "\u00a7": r"\S{}", "\u00a0": "~", "\u2032": r"$'$", "\u00e9": r"\'e", "\u00fc": r'\"u',
    "\u00f6": r'\"o', "\u00e4": r'\"a', "\u00e7": r"\c{c}", "\u00df": r"\ss{}",
}
UNI.update(EXTRA)

ESCAPE = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
          "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}

NOCITE = [False]
TABNO = [0]
CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")

def fix_unicode(t):
    for k, v in UNI.items():
        t = t.replace(k, v)
    return t

def esc(t):
    return "".join(ESCAPE.get(c, c) for c in t)

def cites(t):
    def rep(m):
        nums = [int(x) for x in re.split(r"\s*[,;]\s*", m.group(1)) if x.isdigit()]
        if not nums:
            return m.group(0)
        return r"\cite{" + ",".join("ref%d" % k for k in nums) + "}"
    return CITE.sub(rep, t)

def inline(t):
    math = []

    def keep(m):
        math.append(m.group(0))
        return chr(1) + str(len(math) - 1) + chr(1)

    t2 = re.sub(r"[$][^$]*[$]", keep, t)
    BT = chr(96)
    code = []

    def keepc(m):
        code.append(m.group(1))
        return chr(2) + str(len(code) - 1) + chr(2)

    t2 = re.sub(BT + "([^" + BT + "]+)" + BT, keepc, t2)
    refs = []

    def keepr(m):
        refs.append((m.group(1), m.group(2)))
        return chr(3) + str(len(refs) - 1) + chr(3)

    t2 = re.sub(r"(~?)\\ref\{([^}]+)\}", keepr, t2)
    t2 = esc(t2)
    t2 = fix_unicode(t2)
    t2 = re.sub(r"[*][*](.+?)[*][*]", lambda m: chr(92) + "textbf{" + m.group(1) + "}", t2)
    t2 = re.sub(r"(?<![*])[*]([^*]+)[*](?![*])", lambda m: chr(92) + "emph{" + m.group(1) + "}", t2)
    t2 = re.sub(chr(2) + "(" + chr(92) + "d+)" + chr(2),
                  lambda m: chr(92) + "texttt{" + esc(fix_unicode(code[int(m.group(1))])) + "}", t2)
    t2 = cites(t2)
    t2 = re.sub(chr(1) + "(" + chr(92) + "d+)" + chr(1), lambda m: math[int(m.group(1))], t2)
    t2 = re.sub(chr(3) + "(" + chr(92) + "d+)" + chr(3),
                  lambda m: refs[int(m.group(1))][0] + chr(92) + "ref{" + refs[int(m.group(1))][1] + "}", t2)
    return t2

def parse_pipe(rows):
    head, body = None, []
    for r in rows:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        if head is None:
            head = cells
        else:
            body.append(cells)
    return head or [], body

def col_align(body, ncol):
    align = []
    for j in range(ncol):
        vals = [(r[j] if j < len(r) else "") for r in body]
        numeric = bool(vals) and all((not v) or (re.fullmatch(r"[0-9.,;%+\-\[\]()/\s]+", v) and re.search(r"[0-9]", v)) for v in vals)
        align.append("r" if numeric else "l")
    return "".join(align)

def _brk(t):
    """Allow line breaks inside long unbreakable path-like tokens of a wrapped cell."""
    for a, b in (("/", "/" + BS + "allowbreak{}"),
                 (BS + "_", BS + "_" + BS + "allowbreak{}"),
                 ("-", "-" + BS + "allowbreak{}")):
        t = t.replace(a, b)
    return t


TWOCOL = [False]
WIDECOLS = [5]
AIRY = [False]


CHARW = 0.0116          # 3.95 pt per character at 7 pt over a 341 pt text block
CHARPT = 3.95
TEXTW_ONE = 341.0


def _min_fracs(head, body, ncol):
    """Narrowest width a column can take without breaking its longest token."""
    mins = []
    for j in range(ncol):
        vals = [head[j] if j < len(head) else ""] + [(r[j] if j < len(r) else "") for r in body]
        longest = 0
        for v in vals:
            v = inline(v).replace(BS + "allowbreak{}", "")
            plain = re.sub(r"\\[A-Za-z]+", "", v)
            plain = re.sub(r"[\$\{}\\]", "", plain).strip()
            if plain and not re.search(r"[A-Za-z]", plain):
                longest = max(longest, len(plain))
                continue
            for tok in re.split(r"[\s/-]+", v):
                for ch in ("$", "{", "}"):
                    tok = tok.replace(ch, "")
                longest = max(longest, len(tok))
        mins.append(max(0.05, min(0.34, longest * CHARW + 0.008)))
    return mins


def table_tex(caption, rows, note):
    head, body = parse_pipe(rows)
    ncol = max([len(head)] + [len(r) for r in body]) if (head or body) else 1
    head = head + [""] * (ncol - len(head))
    body = [r + [""] * (ncol - len(r)) for r in body]
    align = col_align(body, ncol)
    _maxlen = max([len(c) for c in head] + [len(c) for r in body for c in r] + [0])
    chars = []
    for j in range(ncol):
        vals = [head[j] if j < len(head) else ""] + [(r[j] if j < len(r) else "") for r in body]
        chars.append(max(len(v) for v in vals))
    natural = sum(chars) * CHARPT + 4.0 * ncol
    wrap = ncol >= 8 or _maxlen > 38 or ((not TWOCOL[0]) and natural > 0.97 * TEXTW_ONE)
    if wrap:
        lens = [max(6.0, min(26.0, ch * 0.95 + 2)) for ch in chars]
        tot = sum(lens)
        fracs = [max(0.06, min(0.30, f / tot)) for f in lens]
        norm = sum(fracs)
        fracs = [f / norm for f in fracs]
        if not TWOCOL[0]:
            need = _min_fracs(head, body, ncol)
            if sum(need) < 0.96:
                extra = 0.96 - sum(need)
                tw = sum(fracs) or 1.0
                fracs = [m + extra * f / tw for m, f in zip(need, fracs)]
            else:
                tw = sum(need)
                fracs = [m / tw * 0.96 for m in need]
        align = "".join(">{\\raggedright\\arraybackslash}p{\\dimexpr %.3f\\textwidth-2\\tabcolsep\\relax}" % f
                        for f in fracs)
    def cell(c):
        t = inline(c)
        return _brk(t) if wrap and len(c) > 18 else t
    if wrap:
        inner = [r"\setlength{\emergencystretch}{1.5em}"]
    else:
        inner = []
    inner += [r"\begin{tabular}{@{}" + align + r"@{}}", r"\toprule",
             " & ".join(cell(c) for c in head) + r" \\", r"\midrule"]
    for r in body:
        inner.append(" & ".join(cell(c) for c in r) + r" \\")
    inner += [r"\bottomrule", r"\end{tabular}"]
    wide = ncol >= WIDECOLS[0] or _maxlen > 40
    env = "table*" if (TWOCOL[0] and wide) else "table"
    out = [r"\begin{" + env + r"}[!t]", r"\centering", r"\caption{" + inline(caption) + "}"]
    short_cells = _maxlen <= 14 and ncol <= 7
    if ncol >= 6 or (env == "table" and _maxlen > 18):
        size, sep = r"\footnotesize", 2
    else:
        size, sep = r"\small", 3
    if AIRY[0] and env == "table*" and short_cells:
        sep = 5
    out.append(size)
    out.append(r"\setlength{\tabcolsep}{%dpt}" % sep)
    out += inner
    out.append(r"\end{" + env + "}")
    if note:
        out += [r"\noindent\footnotesize\textbf{Note:} " + inline(note), r"\normalsize"]
    return "\n".join(out)

def convert(md, figdir, outdir):
    lines = md.replace("\r\n", "\n").split("\n")
    n = len(lines)
    title = None
    out = []
    i = 0
    eqno = 0
    figc = 0
    in_refs = False
    while i < n:
        st = lines[i].strip()
        if st == "---":
            i += 1
            continue
        if lines[i].startswith("# ") and title is None:
            title = st[2:].strip()
            i += 1
            continue
        if st.startswith("## Abstract"):
            j = i + 1
            buf = []
            while j < n and not lines[j].startswith("**Keywords") and not lines[j].startswith("## "):
                buf.append(lines[j].strip())
                j += 1
            out.append(r"\abstract{" + inline(" ".join(x for x in buf if x)) + "}")
            i = j
            continue
        if st.startswith("**Keywords"):
            kw = st.split(":", 1)[1].strip()
            out.append(r"\keywords{" + ", ".join(inline(k.strip()) for k in kw.split(";") if k.strip()) + "}")
            out.append(r"\maketitle")
            i += 1
            continue
        if lines[i].startswith("## "):
            head = re.sub(r"^\d+\.\s*", "", st[3:])
            if in_refs:
                out.append(r"\end{thebibliography}")
                in_refs = False
            if head.strip().lower().startswith("references"):
                in_refs = True
                out.append(r"\begin{thebibliography}{99}")
                i += 1
                continue
            if re.match(r"^Text S\d+", head):
                out.append(r"\section*{" + inline(head) + "}")
                out.append(r"\addcontentsline{toc}{section}{" + inline(head) + "}")
            else:
                out.append(r"\section{" + inline(head) + "}")
            i += 1
            continue
        if lines[i].startswith("### "):
            head = re.sub(r"^\d+(\.\d+)*\s*", "", st[4:])
            if re.match(r"^S\d+\.", head):
                out.append(r"\subsection*{" + inline(head) + "}")
            else:
                out.append(r"\subsection{" + inline(head) + "}")
            i += 1
            continue
        if in_refs:
            m = re.match(r"\[(\d+)\]\s*(.*)", st)
            if m:
                buf = [m.group(2)]
                i += 1
                while i < n and lines[i].strip() and not re.match(r"^\[(\d+)\]", lines[i].strip()):
                    buf.append(lines[i].strip())
                    i += 1
                out.append(r"\bibitem{ref" + m.group(1) + "} " + inline(" ".join(buf)))
                continue
            i += 1
            continue
        if st.startswith("```latex"):
            i += 1
            blk = []
            while i < n and not lines[i].strip().startswith("```"):
                blk.append(lines[i])
                i += 1
            out.append("\n".join(blk))
            i += 1
            continue
        if st.startswith("$$"):
            buf = []
            if st.endswith("$$") and len(st) > 4:
                buf.append(st.strip("$").strip())
                i += 1
            else:
                i += 1
                while i < n and not lines[i].strip().startswith("$$"):
                    buf.append(lines[i].strip())
                    i += 1
                i += 1
            eqno += 1
            body = fix_unicode(" ".join(buf))
            if eqno in (3, 16):
                out.append(r"\begin{equation}\begin{adjustbox}{max width=\linewidth}$\displaystyle "
                           + body
                           + r"$\end{adjustbox}\label{eq:" + str(eqno) + "}\end{equation}")
            else:
                out.append(r"\begin{equation}" + body + r"\label{eq:" + str(eqno) + "}\end{equation}")
            continue
        mo = re.match(r"\*\*(\d+)\.\*\*\s*(.*)", st)
        if mo:
            items = []
            while i < n:
                mi = re.match(r"\*\*(\d+)\.\*\*\s*(.*)", lines[i].strip())
                if not mi:
                    break
                cur = [mi.group(2)]
                i += 1
                while i < n:
                    nxt = lines[i].strip()
                    if (not nxt) or nxt.startswith("#") or nxt.startswith("|") or nxt.startswith("![") \
                       or nxt.startswith("$$") or re.match(r"\*\*\d+\.\*\*", nxt) \
                       or nxt.startswith("**Table") or nxt == "---":
                        break
                    cur.append(nxt)
                    i += 1
                items.append(" ".join(cur))
                while i < n and not lines[i].strip():
                    i += 1
            out.append("\\begin{enumerate}")
            for it in items:
                out.append("\\item " + inline(it))
            out.append("\\end{enumerate}")
            continue
        m = re.match(r"\*\*Table (?:S(\d+)|(\d+))\.\*\*\s*(.*)", st)
        if m:
            snum = m.group(1)
            cap = m.group(3)
            if snum:
                TABNO[0] = max(TABNO[0], int(snum))
            j = i + 1
            rows = []
            while j < n and not lines[j].strip():
                j += 1
            while j < n and lines[j].strip().startswith("|") and lines[j].count("|") >= 3:
                rows.append(lines[j])
                j += 1
                while j < n and not lines[j].strip():
                    j += 1
            note = None
            k = j
            while k < n and not lines[k].strip():
                k += 1
            if k < n and lines[k].strip().startswith("**Note"):
                note = re.sub(r"^\*\*Note:?\*\*\s*", "", lines[k].strip())
                j = k + 1
                while j < n:
                    ns = lines[j].strip()
                    if (not ns or ns.startswith("#") or ns.startswith("|") or ns.startswith("![")
                            or ns.startswith("**Table") or ns.startswith("$$")
                            or ns.startswith("* ") or ns.startswith("- ") or ns == "---"):
                        break
                    note += " " + ns
                    j += 1
            _tb = table_tex(cap, rows, note)
            if snum:
                _tb = ("\\begingroup\\renewcommand{\\thetable}{S" + snum + "}\n"
                       + _tb + "\n\\endgroup")
            out.append(_tb)
            i = j
            continue
        if st.startswith("|") and st.count("|") >= 3:
            rows = []
            j = i
            while j < n and lines[j].strip().startswith("|") and lines[j].count("|") >= 3:
                rows.append(lines[j])
                j += 1
            TABNO[0] += 1
            out.append(table_tex("Supplementary table S%d" % TABNO[0], rows, None))
            i = j
            continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", st)
        if m:
            figc += 1
            _alt = m.group(1)
            _onecol = "|1col" in _alt
            base = os.path.basename(m.group(2))
            src = os.path.join(figdir, base)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(outdir, base))
            cap = ""
            j = i + 1
            k = j
            while k < n and not lines[k].strip():
                k += 1
            snum = None
            if k < n:
                mf = re.match(r"\*\*Figure (?:(S)(\d+)|\d+)\.\*\*", lines[k].strip())
                if mf:
                    snum = mf.group(2) if mf.group(1) else None
                    cap = re.sub(r"^\*\*Figure (?:S\d+|\d+)\.\*\*\s*", "", lines[k].strip())
                    j = k + 1
            _env = "figureS" if _onecol else "figure"
            _w = "\\columnwidth" if _onecol else "\\textwidth"
            _fig = ("\\begin{" + _env + "}[!t]\n\\centering\n\\includegraphics[width=" + _w + "]{" + base +
                    "}\n\\caption{" + inline(cap) + "}\n\\label{fig:" + (("S" + snum) if snum else str(figc)) + "}\n\\end{" + _env + "}")
            if snum:
                _fig = ("\\begingroup\\renewcommand{\\thefigure}{S" + snum + "}\n" + _fig + "\n\\endgroup")
            out.append(_fig)
            i = j
            continue
        if st.startswith("* ") or st.startswith("- "):
            out.append(r"\begin{itemize}")
            cur = None
            while i < n:
                piece = lines[i].strip()
                if not piece:
                    break
                if piece.startswith("* ") or piece.startswith("- "):
                    if cur is not None:
                        out.append(r"\item " + inline(" ".join(cur)))
                    cur = [piece[2:].strip()]
                    i += 1
                    continue
                if (piece.startswith("|") or piece.startswith("#") or piece.startswith("![")
                        or piece.startswith("$$") or piece.startswith("**Table") or piece == "---"):
                    break
                if cur is None:
                    break
                cur.append(piece)
                i += 1
            if cur:
                out.append(r"\item " + inline(" ".join(cur)))
            out.append(r"\end{itemize}")
            continue
            continue
        if not st:
            i += 1
            continue
        buf = [st]
        i += 1
        while i < n:
            nxt = lines[i]
            ns = nxt.strip()
            if (not ns or nxt.startswith("#") or (ns.startswith("|") and ns.count("|") >= 3) or ns.startswith("![")
                    or ns.startswith("**Table") or ns.startswith("**Figure") or ns.startswith("$$")
                    or ns.startswith("* ") or ns.startswith("- ") or ns == "---"):
                break
            buf.append(ns)
            i += 1
        out.append(inline(" ".join(buf)))
        out.append("")
    if in_refs:
        out.append(r"\end{thebibliography}")
    return title or "Untitled", "\n\n".join(x for x in out if x)

PREAMBLE = r"""\documentclass[sn-basic,Numbered,pdflatex]{sn-jnl}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\newtheorem{proposition}{Proposition}
\usepackage{mathrsfs}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{manyfoot}
\usepackage{array}
\usepackage{adjustbox}
\usepackage{booktabs}
\usepackage{geometry}
\makeatletter\providecommand\headerps@out[1]{}\makeatother
\providecommand{\bibfont}{\small}
\renewcommand{\bibfont}{\footnotesize}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\floatsep}{9pt plus 2pt minus 2pt}
\setlength{\intextsep}{9pt plus 2pt minus 2pt}
\setlength{\abovedisplayskip}{4pt plus 1pt minus 1pt}
\setlength{\belowdisplayskip}{4pt plus 1pt minus 1pt}
\setlength{\abovedisplayshortskip}{2pt}
\setlength{\belowdisplayshortskip}{2pt}
\setlength{\abovecaptionskip}{4pt}
\setlength{\belowcaptionskip}{0pt}
\begin{document}
"""

def split_front(md):
    lines = md.replace("\r\n", "\n").split("\n")
    title, abstract, keywords, body_start = None, "", "", None
    i, n = 0, len(lines)
    while i < n:
        st = lines[i].strip()
        if st.startswith("# ") and title is None:
            title = st[2:].strip()
            i += 1
            continue
        if st.startswith("## Abstract"):
            j = i + 1
            buf = []
            while j < n and not lines[j].startswith("**Keywords") and not lines[j].startswith("## "):
                buf.append(lines[j].strip())
                j += 1
            paras, cur = [], []
            for x in buf:
                if x:
                    cur.append(x)
                elif cur:
                    paras.append(" ".join(cur)); cur = []
            if cur:
                paras.append(" ".join(cur))
            abstract = (chr(10) + chr(10)).join(paras)
            i = j
            continue
        if st.startswith("**Keywords"):
            keywords = st.split(":", 1)[1].strip().lstrip("*").strip()
            i += 1
            continue
        if st.startswith("## ") and title is not None:
            body_start = i
            break
        i += 1
    return (title or "Untitled"), abstract, keywords, body_start, lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--out", default="drafts/submission")
    ap.add_argument("--name", default="main")
    ap.add_argument("--figdir", default="drafts/figs")
    ap.add_argument("--esm", action="store_true")
    ap.add_argument("--iicol", action="store_true")
    ap.add_argument("--nocite", action="store_true")
    ap.add_argument("--widecols", type=int, default=5)
    ap.add_argument("--airy", action="store_true")
    ap.add_argument("--textblock", default="")
    a = ap.parse_args()
    NOCITE[0] = a.nocite
    TWOCOL[0] = a.iicol
    WIDECOLS[0] = a.widecols
    AIRY[0] = a.airy
    os.makedirs(a.out, exist_ok=True)
    md = open(a.md, encoding="utf-8", newline="").read()
    title, abstract, keywords, body_start, _ = split_front(md)
    body_md = "\n".join(split_front(md)[4][body_start:])
    front_paras = []
    if a.esm or a.name.lower() == "esm":
        raw = md.replace(chr(13)+chr(10), chr(10)).split(chr(10))
        head = raw[1:body_start]
        keep, skip = [], False
        for l in head:
            s = l.strip()
            if s.startswith('## Abstract'):
                skip = True
                continue
            if s.startswith('**Keywords'):
                skip = False
                continue
            if skip or s.startswith('## '):
                continue
            keep.append(l)
        cur = []
        for l in keep:
            if l.strip():
                cur.append(l.strip())
            elif cur:
                front_paras.append(' '.join(cur)); cur = []
        if cur:
            front_paras.append(' '.join(cur))
    # front_paras: ESM-only lead paragraphs between the title and the first heading
    _, body = convert(body_md, a.figdir, a.out)
    front = []
    front.append("\\title{" + inline(title) + "}")
    front.append(r"\author*[1]{\fnm{Anonymous}\sur{Author(s)}}\affil*[1]{\orgname{Anonymous Institution}}")
    if abstract:
        front.append("\\abstract{" + inline(abstract) + "}")
    if keywords:
        kws = [k.strip() for k in keywords.split(";") if k.strip()]
        front.append("\\keywords{" + ", ".join(inline(k) for k in kws) + "}")
    front.append(r"\maketitle")
    for p in front_paras:
        front.append(inline(p))
    if a.iicol:
        body = (body.replace(r"\begin{figure}[!t]", r"\begin{figure*}[!t]")
                    .replace(r"\end{figure}", r"\end{figure*}"))
    body = (body.replace(r"\begin{figureS}[!t]", r"\begin{figure}[!t]")
                .replace(r"\end{figureS}", r"\end{figure}"))
    extra = ""
    if a.textblock:
        w, h, t = a.textblock.split(",")
        extra = (BS + "geometry{text={%smm,%smm},top=%smm}" + chr(10) + BS + "raggedbottom" + chr(10)) % (w, h, t)
    tex = (PREAMBLE.replace("\\begin{document}", extra + "\\begin{document}") if extra else PREAMBLE) + "\n".join(front) + "\n" + body + "\n\\end{document}\n"
    tex = tex.replace("\\end{itemize}\n\n\\begin{itemize}\n", "")
    if a.iicol:
        tex = tex.replace(r"\documentclass[sn-basic,Numbered,pdflatex]{sn-jnl}",
                          r"\documentclass[sn-basic,Numbered,iicol,pdflatex]{sn-jnl}")
    p = os.path.join(a.out, a.name + ".tex")
    open(p, "w", encoding="utf-8", newline="\n").write(tex)
    print("wrote", p, len(tex), "bytes")
    print("equations", tex.count(r"\begin{equation}"), "tables", tex.count(r"\begin{table}"),
          "figures", tex.count(r"\begin{figure}"), "refs", tex.count(r"\bibitem"))


if __name__ == "__main__":
    main()
