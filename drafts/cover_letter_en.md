# Cover letter

Dear Editor,

We submit our manuscript **"The cost curve of 8 bit in-place weight projection: interval, overwrite and stochastic rounding"** for consideration as a research article.

Deployment-oriented quantization-aware training (QAT) usually overwrites weights in place onto the int8 grid during training and keeps an EMA shadow of them. The projection interval and the moment at which the shadow is written are tuned in practice largely by folklore, and the metrics used to judge them are often the ones that cannot carry the information. Our paper measures those choices under one fixed recipe and reports four things.

1. **A dose curve.** Against switching projection off, deployment MAE is 0.0047 higher at interval 20 and 0.0185 higher at interval 1 (1,556 images that took no part in epoch selection, eight initial seeds, paired, run as the unit of inference). Switching projection off is the best weight-only 8 bit setting we found, on the main configuration and on the vendor NPU chain.
2. **Attribution and a zero-cost fix.** The cost belongs to the overwrite step, not to resting on the grid: fake-quantizing at every step without overwriting (STE) is indistinguishable from switching projection off (−0.0001), and replacing nearest rounding by stochastic rounding at each overwrite removes essentially all of the cost (0.0183 → 0.00004) with the grid, the interval and the training budget unchanged.
3. **An equivalence boundary.** The averaging order moves the exported weights by 4.3–9.1 % in relative L2 and the displacement survives export-time requantization, yet the function-side effect lies below our pre-specified 0.0016 equivalence margin (MDE 0.000383). We report that as a bound that the design can certify, not as a proof that the two orders are equivalent.
4. **An executable protocol.** Five rules for evaluating 8 bit deployment training, including one premise we state explicitly rather than as a discovery: when the training projection and the deployment quantizer share the grid, export-time requantization is idempotent, so 8 bit fidelity metrics such as the quantization cosine carry no information by construction, and an evaluation has to report the task metric measured after deployment quantization.

The interval conclusion is additionally checked on the vendor NPU toolchain, on a second backend (OpenVINO with NNCF) and on two extrapolation axes (a ResNet-18 architecture and the KonIQ-10k dataset); we report those as directional observations and do not present them as independent replications.

The paper does not propose a new QAT algorithm, and we do not claim stochastic rounding or master-copy-free training as our own: both are cited from the literature, and our contribution is a controlled measurement of their effect inside one recipe plus a reproducible evaluation protocol.

Code, scripts, derived artefacts and the mapping from every claim to its artefact are available in the anonymised repository https://github.com/una393/utkface-snap-order. The datasets and the trained weights are not redistributed, because of upstream licence terms (see the NOTICE file in that repository).

Sincerely,
Anonymous Author(s)
Anonymous Institution
