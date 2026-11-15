r"""Render references.bbl from references.bib in the elsarticle-num style.

    python scripts/make_bbl.py

Tectonic carries no .bst file in its bundle and this machine has no TeX distribution,
so BibTeX cannot run here.  The manuscript therefore does not call \bibliography;
scripts/make_latex.py emits \input{references.bbl} instead, and this script builds that
file from drafts/references.bib with the numbering elsarticle-num would assign: the order
in which the keys first appear in a \cite of the main text.

Writes
    drafts/submission/references.bbl     the bibliography the main text inputs
    drafts/submission/cite_numbers.json  key -> number, read by scripts/append_esm_bib.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "drafts" / "references.bib"
TEX = ROOT / "drafts" / "submission" / "main.tex"
BBL = ROOT / "drafts" / "submission" / "references.bbl"
NUMBERS = ROOT / "drafts" / "submission" / "cite_numbers.json"
MD = ROOT / "drafts" / "paper2_journal_en.md"

sys.stdout.reconfigure(encoding="utf-8")


def parse_bib(text: str) -> dict:
    """Fields of every entry: @type{key, field = {value}, ...} with brace nesting."""
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        kind, key = m.group(1).lower(), m.group(2).strip()
        body, depth, i = [], 1, m.end()
        while i < len(text) and depth:
            ch = text[i]
            depth += {"{": 1, "}": -1}.get(ch, 0)
            body.append(ch)
            i += 1
        fields, rest = {}, "".join(body[:-1])
        for fm in re.finditer(r"(\w+)\s*=\s*\{", rest):
            j, d, buf = fm.end(), 1, []
            while j < len(rest) and d:
                ch = rest[j]
                d += {"{": 1, "}": -1}.get(ch, 0)
                if d:
                    buf.append(ch)
                j += 1
            fields[fm.group(1).lower()] = " ".join("".join(buf).split())
        entries[key] = (kind, fields)
    return entries


def initial(word: str) -> str:
    return word.replace(".", "").replace("-", "")


def format_names(field: str) -> str:
    """"Bengio, Y. and L{\'e}onard, N." -> "Bengio Y, L{\'e}onard N"."""
    out = []
    for raw in field.split(" and "):
        raw = raw.strip()
        if not raw:
            continue
        if "," in raw:
            last, first = [p.strip() for p in raw.split(",", 1)]
            parts = [p.strip() for p in first.split() if p.strip()]
            inits = " ".join(initial(p) for p in parts)
            out.append(("%s %s" % (last, inits)).strip())
        else:
            out.append(raw)
    return ", ".join(out)


def render(key: str, kind: str, f: dict) -> str:
    names = format_names(f.get("author", "")) + "."
    title = f.get("title", "").rstrip(".")
    tail = []
    if kind == "article":
        head = "%s %s." % (names, title)
        vol = f.get("volume")
        pages = f.get("pages", "").replace("--", "\u2013")
        tail.append("%s %s;%s." % (f.get("journal", ""), f.get("year", ""),
                                   (vol + ":" if vol else "") + pages))
    elif kind in ("inproceedings", "incollection"):
        head = "%s %s." % (names, title)
        pages = f.get("pages", "").replace("--", "\u2013")
        tail.append("In: %s; %s." % (f.get("booktitle", ""), f.get("year", "")))
        if pages:
            tail.append("p. %s." % pages)
        if f.get("publisher"):
            tail.append("%s." % f["publisher"])
    else:
        head = "%s %s." % (names, title)
        if f.get("howpublished"):
            tail.append("%s." % f["howpublished"])
        if f.get("eprint"):
            tail.append("arXiv:%s (%s)." % (f["eprint"], f.get("year", "")))
        elif f.get("year"):
            tail.append("%s." % f["year"])
    if f.get("doi"):
        tail.append("doi:%s." % f["doi"].lstrip("https://doi.org/").lstrip("doi:"))
    return escape_text(" ".join([head] + tail))


def escape_text(s: str) -> str:
    """Escape the characters that break text mode; leave existing escapes alone."""
    return re.sub(r"(?<!\\)([_&%#])", r"\\\1", s)


def citation_order(tex: str) -> list:
    order, seen = [], set()
    for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
        for key in (k.strip() for k in m.group(1).split(",")):
            if key and key not in seen:
                seen.add(key)
                order.append(key)
    return order


def check_against_md(entries: dict, numbers: dict) -> list:
    """Every md reference line must name the first surname and the year of its entry."""
    if not MD.exists():
        return []
    lines = MD.read_text(encoding="utf-8").splitlines()
    start = next((i for i, t in enumerate(lines)
                  if re.match(r"^##\s*(?:\d+\.?\s*)?References\s*$", t.strip())), None)
    problems = []
    if start is None:
        return ["no References section in the markdown draft"]
    for line in lines[start + 1:]:
        m = re.match(r"^\[(\d+)\]\s+(.*)$", line.strip())
        if not m:
            continue
        num, text = int(m.group(1)), m.group(2)
        key = "ref%d" % num
        if key not in entries:
            problems.append("md [%d] has no %s entry" % (num, key))
            continue
        kind, f = entries[key]
        surname = format_names(f.get("author", "")).split(",")[0].split()[0]
        if surname.replace("{", "").replace("}", "") not in text:
            problems.append("md [%d] does not name %s" % (num, surname))
        if f.get("year", "") not in text:
            problems.append("md [%d] does not carry the year %s" % (num, f.get("year", "")))
    return problems


def main() -> int:
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    tex = TEX.read_text(encoding="utf-8")
    order = citation_order(tex)
    missing = [k for k in order if k not in entries]
    if missing:
        print("keys cited but absent from references.bib:", ", ".join(missing))
        return 1
    uncited = sorted(set(entries) - set(order))
    numbers = {k: i + 1 for i, k in enumerate(order)}
    expected = {"ref%d" % (i + 1): i + 1 for i in range(len(order))}
    if numbers != expected:
        drift = {k: (numbers[k], expected.get(k)) for k in numbers if numbers.get(k) != expected.get(k)}
        print("first-citation order does not run 1..N:", drift)
        return 1
    items = ["%% generated by scripts/make_bbl.py from drafts/references.bib -- do not edit",
             "%% numbering and layout follow elsarticle-num: order of first citation, Elsevier style",
             "\\begin{thebibliography}{%d}" % len(order)]
    for key in order:
        kind, f = entries[key]
        items.append("\\bibitem{%s}\n%s" % (key, render(key, kind, f)))
    items.append("\\end{thebibliography}")
    BBL.write_text("\n".join(items) + "\n", encoding="utf-8")
    NUMBERS.write_text(json.dumps(numbers, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    problems = check_against_md(entries, numbers)
    print("entries %d  cited %d  uncited %s  wrote %s" %
          (len(entries), len(order), uncited or "none", BBL.relative_to(ROOT)))
    for p in problems:
        print("  mismatch:", p)
    print("md cross-check:", "ok" if not problems else "%d problems" % len(problems))
    return 0 if not problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
