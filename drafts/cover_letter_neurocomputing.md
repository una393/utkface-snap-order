# Cover letter — Neurocomputing

[Date]

[FILL IN: corresponding author name, full affiliation and postal address, active e-mail, ORCID]

Dear Editor-in-Chief and Editorial Board of *Neurocomputing*,

Please consider the enclosed manuscript, **"A Run-Level Evaluation Protocol for In-Place Weight Projection in 8-bit Quantization-Aware Training"**, for publication as an Original Research Article. We intend the subscription (non-open-access) route.

**Why Neurocomputing.** The journal calls for fundamental contributions spanning neurocomputing theory, practice and applications. This work sits at their intersection. The operators we study — a symmetric per-output-channel int8 snap, an EMA shadow and a projection interval — are existing practice; the contribution is the first joint audit of their composition under one fixed recipe, an admissibility condition that decides when a deployment fidelity metric is informative, and a run-level inferential framework that certifies both positive and null results. On the practice and applications side, the protocol is exercised on a compiled int8 chain for a target NPU, with the simulator checked against the board on 800 inferences, and replicated on ONNX Runtime and OpenVINO.

**What is new, and what is not.** We do not propose a new quantizer, loss or architecture, and we state this explicitly in the paper. What is new is:

1. a measured cost curve for in-place projection: +0.0185 MAE at every step and +0.0047 at every 20 steps against switching projection off, in all eight paired runs, and it survives Holm correction within the six-test family;
2. an attribution backed by two controls — a straight-through estimator that fake-quantizes without overwriting shows no cost, while stochastic rounding at each overwrite reduces the cost from 0.0183 to 0.00004 at 8 bits, with grid, interval and budget unchanged;
3. Proposition 1 and Algorithm 1, which give a checkable condition under which export quantization is idempotent and a centered cosine is constant by construction, so that a saturated fidelity score must not be quoted;
4. a five-rule decision table released with the scripts and artifacts that regenerate every number.

Table S1 maps six dimensions against the closest published designs (search cutoff 2026-11-02); none covers all six. Accordingly this is not a manuscript that merely applies standard methods: it supplies the criterion, the check and the evidence base that deployment reports typically lack.

**Why the evidence supports the claims.** The unit of inference is the run (eight paired initial seeds on one fixed split), scored on 1,556 images that took no part in epoch selection. Six confirmatory contrasts were fixed in the analysis plan before the run-level readings were computed and carry Holm-adjusted p-values as one family. The null result on averaging order is reported as an equivalence test with a pre-specified 0.0016 margin, its provenance, the 80% minimum detectable effect and a margin sensitivity analysis, rather than as "no difference". Everything else is labelled exploratory. Duplicate-photograph leakage, the optimistic 400-image basis, the non-decidability of the 4-bit ranking and the limits of the external axes are quantified in the text and supplement.

**Relation to the Software Track.** If the Editors consider the contribution better suited to it, the admissibility check and the decision table can be reformatted as an Original Software Publication: the training and export code, all statistics scripts and a per-file SHA-256 manifest are already in the anonymized repository (674 files, about 11 MB, MIT license for the code), and Algorithm 1 specifies the checker entry point.

**Statements.** All authors approved the manuscript. The work is original, is not under consideration elsewhere, and no part of it has been published. The authors declare no competing financial or personal interests. No external funding was received. Data are not redistributed because of upstream licence terms (UTKFace: non-commercial research purposes only); the exact splits, derived results and all code are available in the anonymized repository https://github.com/una393/utkface-snap-order, and Table S13 maps every claim to its artifact and script. The study is a secondary analysis of two public benchmark datasets with no new human data collected; ethics and broader-impact limitations are stated explicitly in the dedicated section.

[FILL IN: 3–5 suggested reviewers with affiliations and e-mails, or delete this line if the submission system collects them separately.]

Thank you for considering the manuscript.

Sincerely,

[FILL IN: corresponding author, on behalf of all authors]
