# -*- coding: utf-8 -*-
"""Assemble the journal-length English manuscript and the ESM from the r37 parts."""
import io, os, re

def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")

src = rd("drafts/paper2_snap_order.md")
lines = src.split("\n")
title = lines[0]
refs_start = next(i for i, l in enumerate(lines) if l.startswith("## 11. References"))
refs = "\n".join(lines[refs_start:]).rstrip() + "\n"

ABSTRACT = ("Deployment-oriented quantization-aware training often projects weights in place onto an int8 grid during "
"training and maintains an EMA shadow, yet how the projection interval, the moment at which the shadow is written and "
"the averaging order change deployment metrics lacks a controlled measurement. On UTKFace age regression we fix the "
"recipe (symmetric per-output-channel int8, 20 epochs, five configuration sequences of 8 initial seeds, a fixed split, "
"the run as the unit of inference), sweep the interval, add the endpoint at which projection is switched off, and "
"attribute the cost with two zero-cost controls. Against switching projection off, deployment MAE is 0.0047 and 0.0185 "
"higher at intervals 20 and 1 (1,556 images that took no part in epoch selection; 0.0060 and 0.0214 at 6 bits, "
"exploratory and not corrected across precisions); the straight-through control, which fake-quantizes at every step "
"without overwriting, detects no difference from switching projection off, while replacing nearest rounding by "
"stochastic rounding at each overwrite moves the same cost from 0.0183 to 0.00004 with the grid, the interval and the "
"budget unchanged. Writing the shadow brings an improvement of 0.0014; the averaging order is excluded only against the "
"pre-specified equivalence bound of 0.0016 (MDE 0.000383), which is not a proof of equivalence. One caution follows: "
"when training and deployment share the grid, export-time requantization is idempotent by construction, so 8 bit "
"fidelity metrics carry no information under this convention. These decisions are organised into five executable "
"evaluation rules (SS6.1). Conclusions are limited to a single dataset, a single model and a fixed budget; the "
"cross-backend replication and the extrapolation axes are directional observations only.")

KEYWORDS = "quantization-aware training; weight projection; deployment evaluation; stochastic rounding; equivalence testing"

DECL = """## Declarations

### Funding

This work received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

### Competing interests

The authors declare that they have no competing interests.

### Ethics approval and consent to participate

Not applicable. The study is a secondary analysis of two publicly available benchmark datasets (UTKFace and KonIQ-10k); no participants were recruited and no new human data were collected.

### Consent for publication

Not applicable.

### Data availability

The UTKFace and KonIQ-10k datasets are publicly available from their original distributors and are not redistributed here because of their licence terms; the exact files, splits and checkpoints used are recorded in the anonymized repository https://github.com/una393/utkface-snap-order, together with the scripts that regenerate every reported number. Table S1 of the Electronic Supplementary Material maps each claim to its artifact.

### Code availability

All training, quantization, compilation and analysis code, and the derived result files, are available in the same anonymized repository (595 files, about 11 MB). The code is released under the MIT licence; data and weights follow the upstream non-commercial terms, as stated in the NOTICE file.

### Authors' contributions

All authors designed the study, carried out the experiments and wrote the manuscript.
"""

def part(name):
    return rd("drafts/_r37/part_%s.md" % name).rstrip() + "\n"

def esm(name):
    return rd("drafts/_r37/esm_%s.md" % name).rstrip() + "\n"

main = [title, "", "Anonymous Author(s)", "Anonymous Institution", "", "## Abstract", "", ABSTRACT, "",
        "**Keywords:** " + KEYWORDS, "", part("A"), part("B"), "## 5. Results", "", part("C"), part("D"), part("E"),
        part("F"), refs, DECL]

md = "\n".join(main)
# ESM pointer renaming: TA..TF -> S1..S6
for a, b in zip("ABCDEF", "123456"):
    md = md.replace("Text S" + a, "Text S" + b)
    md = md.replace("S" + a + ".", "S" + b + ".")
# the artifact map moves to the ESM as Table S1
md = md.replace("Table 16", "Table S1")
md = md.replace("\u00a7\u00a7", "\u00a7")
io.open("drafts/paper2_journal_en.md", "w", encoding="utf-8", newline="\n").write(md)

blocks = [esm(x) for x in "ABCDEF"]
esm_md = "# Electronic Supplementary Material\n\n" + "\n".join(blocks)
for a, b in zip("ABCDEF", "123456"):
    esm_md = esm_md.replace("Text S" + a, "Text S" + b)
    esm_md = esm_md.replace("S" + a + ".", "S" + b + ".")
esm_md = esm_md.replace("**Table 16.**", "**Table S1.**").replace("Table 16", "Table S1")
io.open("drafts/paper2_esm_en.md", "w", encoding="utf-8", newline="\n").write(esm_md)

def words(t):
    t = "\n".join(l for l in t.split("\n") if not l.startswith("| ") and not l.startswith("$$")
                  and not l.startswith("!["))
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", t))

print("main words", words(md))
print("esm words", words(esm_md))
print("tables", md.count("**Table "), "figs", md.count("!["), "eqs", md.count("$$") // 2, "refs", md.count("\n[") + md.count("^\n["))
