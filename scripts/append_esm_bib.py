# -*- coding: utf-8 -*-
"""Resolve the ESM citations to main-text reference numbers.

The ESM quotes the same 44 references as the main text, so it repeats no
bibliography: each \\cite{refN} becomes the literal bracket number that the
main text assigns to that key.
"""
import io, json, re

# scripts/make_bbl.py owns the key -> number mapping; the markdown list is the fallback.
try:
    mapping = {k: int(v) for k, v in json.load(io.open("drafts/submission/cite_numbers.json", encoding="utf-8")).items()}
    print("refs found", len(mapping), "from cite_numbers.json")
except FileNotFoundError:
    main = io.open("drafts/paper2_journal_en.md", encoding="utf-8", newline="").read().replace("\r\n", "\n")
    m_ref = re.search(r"^##\s*(?:\d+\.?\s*)?References\s*$", main, re.M)
    if m_ref is None:
        raise SystemExit("references heading not found")
    order = []
    for ln in main[m_ref.start():].split("\n")[1:]:
        m = re.match(r"^\[(\d+)\]\s*(.*)$", ln)
        if m:
            order.append(m.group(1))
    if not order:
        raise SystemExit("no reference entries found")
    mapping = {"ref" + num: num for num in order}
    print("refs found", len(order), "from the markdown draft")

tex = io.open("drafts/submission/esm.tex", encoding="utf-8", newline="").read()
tex = re.sub(r"\n\\section\*\{References\}.*?(?:\\end\{thebibliography\}|\\input\{references\.bbl\}[^\n]*)\n?",
              "\n", tex, flags=re.S)

seen = {}


def repl(m):
    keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
    nums = []
    for k in keys:
        if k not in mapping:
            raise SystemExit("unknown cite key in the ESM: " + k)
        seen[k] = seen.get(k, 0) + 1
        nums.append(mapping[k])
    return "[" + ", ".join(str(x) for x in nums) + "]"


tex, n = re.subn(r"\\cite\{([^}]*)\}", repl, tex)
if "\\cite" in tex:
    raise SystemExit("a citation form other than \\cite survives")
io.open("drafts/submission/esm.tex", "w", encoding="utf-8", newline="\n").write(tex)
print("citations resolved", n, "distinct keys", len(seen))
