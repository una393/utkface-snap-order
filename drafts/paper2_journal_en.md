# A Run-Level Evaluation Protocol for In-Place Weight Projection in 8-bit Quantization-Aware Training

Anonymous Author(s)
Anonymous Institution

## Abstract

Int8 weights reach a compiler from the last steps of a conventional training loop: many recipes project weights onto the deployment grid, and most export an exponential moving average (EMA). Which tensor is exported, and how often projection runs, decide what a reported accuracy describes; both are treated as implementation details, and the export is scored by a fidelity number that saturates at 1.0000 and cannot rank checkpoints.

We evaluate both at run level on UTKFace age regression under one int8 recipe, with a straight-through control that quantizes the forward pass without overwriting and a stochastic-rounding variant holding grid, interval and budget fixed. A resolution bound (Proposition 1) and an executable pre-check (Algorithm 1) turn saturation into a decision rule, applied to an NPU compiler, two runtimes and CIFAR-10.

Against projection disabled, FP32 deployment mean absolute error (MAE) rises by 0.0185 when every step is projected and 0.0047 at interval 20. The cost originates in the overwrite, not the grid: the straight-through control shows none, stochastic rounding lowers it from 0.0183 to 0.00004 at unchanged grid, interval and budget, and CIFAR-10 loses 5.0 top-1 points. Averaging order changes MAE by 0.00003 [−0.00025, +0.00031], within a pre-specified 0.0016 bound, and the EMA shadow lowers it by 0.0014. Conclusions cover one dataset and model.

**Keywords:** quantization-aware training; int8 deployment; weight projection; run-level evaluation; stochastic rounding; equivalence testing

## 1. Introduction

A quantized model is described by its architecture and its recipe, yet what a compiler receives is a single tensor chosen in the last steps of training. Two decisions in that loop fix it: how often the weights are pushed onto the deployment grid, and whether the average exported is a running one. Both are usually left as implementation detail, and the tensor is scored by a fidelity number that the loop itself can pin. This paper isolates the two decisions and measures each on the task metric after export quantization.

We call one push a *projection*: every quantizable weight is replaced by the nearest point of the deployment grid, at the per-output-channel scale of §3.2. It can run after every optimizer step or every $N$ steps, and $N = \infty$ switches it off; most loops also keep an exponential moving average of the iterates and export that average instead of the last one. Each mechanism is well studied; their composition is not. Because $S(E[W]) \ne E(S[W])$, averaging before projection gives different weights from projecting before averaging, and the two exports differ by several percent in relative L2 distance.

Which of these settings should be shipped is a practical question, and the scores usually reported cannot answer it. When the training projection and the deployment quantizer share operator and rounding rule, nothing clips, and the tensor already sits on the training grid, requantizing changes nothing and a centered cosine reads 1.0000 by construction — a constant cannot rank two checkpoints. Our chains recompute the scale at export, so the identity becomes a bound: Proposition~\ref{prop:noinfo} states the residual condition under which the printed score is fixed, and Algorithm~\ref{alg:precheck} applies it before any fidelity number is quoted.

We therefore audit the chain rather than a single number: one fixed recipe with three knobs — how often the projection runs, whether a shadow is kept, and the order of projection and averaging. Each setting is scored after export quantization on images that took no part in choosing the epoch; the run is the unit of inference, eight paired seeds share one split, and the chain is compiled for a target NPU and repeated on two open-source runtimes.

Projection is existing practice: SWALP, SQWA, BinaryRelax, DQT and ECO all write quantized weights during training (Table S1), and none treats the interval, the shadow and the ordering as variables or audits the fidelity metric quoted beside the export.

The paper is an executable protocol, not a new quantizer: the operator is existing practice; the new material is the joint audit, the admissibility check and validation on a real deployment chain. Three results support it:

* **A measured cost, in a setting we state.** Projecting more often raises deployment mean absolute error (MAE): against projection off, every step costs 0.0185 MAE (about 1.85 years of age) and every 20 steps 0.0047, in all eight paired runs and after Holm correction. Switching projection off is then the best weight-only 8-bit setting under this recipe; how often shipped toolchains project in place is not something we establish here.
* **An attribution and a zero-cost fix.** The cost follows the in-place overwrite, not the grid: a control that fake-quantizes every step without overwriting shows no cost, and stochastic rounding removes almost all of it (0.0183 to 0.00004) with grid, interval and budget unchanged. Neither control measures the vanished updates directly, so the attribution is supported, not proved.
* **An executable five-rule evaluation protocol.** The rules fix when a fidelity score may be quoted, why a per-node cosine cannot rank models, which setting to drop first at 8 bits, and how to report a null result with a stated margin. Algorithm 1 makes the first check runnable, Table 6 turns the five into a decision table, and Table S13 maps each rule to its artifact.

The scope is deliberately narrow: the pre-specified conclusions rest on one dataset, one architecture and one budget. A second network and a second dataset reproduce the direction of the interval effect but not its magnitude; the order contrast on those axes, the cross-backend checks and the 4-bit ranking stay directional or cannot be decided. What transfers is the protocol rather than a new quantizer.

## 2. Related work

Quantization-aware training widely uses straight-through estimation, popularized by Bengio et al. [1]. PACT learned the activation clipping threshold [2], LSQ the step size [3], and INQ raises the fraction of quantized weights in stages [4]. Low-precision training distinguishes keeping a high-precision master and fake-quantizing the forward pass from updating low-precision weights directly; under nearest rounding the second erases any update below half a grid step, which stochastic rounding and error feedback exist to repair [5], [6].

Integer-inference practice tied simulated to deployment arithmetic: Jacob et al. run inference in integer arithmetic only and align the simulation with it [7], Krishnamoorthi [8] and Nagel et al. [9] make quantize/dequantize (QDQ) placement a design surface, and Lee et al. model the quantizer transition as a scheduled rate [10], which we compare with our interval $N$ in Text S6.1, item 5, and Table S12.

Iterate averaging [11] and its exponential variant are standard practice, and Mean Teacher [12] popularized the EMA-shadow formulation. SWALP [13] and SQWA [14] average under quantization but average checkpoints or the trajectory; SWA [15] and model soups [16] rest on separately trained models. The shadow studied here averages adjacent iterates inside one run and costs no extra training.

Reported "quantization error" is often a cosine or a relative-L2 distance between float and quantized outputs. Such a metric stops separating good models from bad ones once it saturates, and saturation depends on whether a projection ran during training, not on deployment quality. MQBench makes deployability a reproducible benchmark tied to concrete implementations [17]; our scope is narrower, changing only the criterion to the task metric after explicit quantization. Low-bit methods change the quantizer or the basis it acts on: the per-channel rescaling of SmoothQuant [18], the learned rounding of AdaRound [19] and BRECQ [20], the Hessian weighting of HAWQ and HAWQ-V2 [21, 22], the layer-wise schemes of GPTQ [23] and AWQ [24], and fixed or learned rotations [25], [26]. Here both are held fixed over the whole sweep.

Every operator in our recipe is taken from existing practice: the projection is the plain symmetric per-output-channel snap, the only weight-only member of that family, with no learned scale; the shadow is a standard exponential average; and the interval is swept rather than designed. The one setting that does affect the outcome is also borrowed: replacing nearest with stochastic rounding recovers 99.8 % of the measured cost on average (94.4 % to 105.9 % per seed), and that rule comes from [5], as does the work that motivates overwriting [6]. Two published readings of the same recipe differ in the rounding mode, so we report the rounding mode, the interval and the evaluation basis next to every number.

Whether a study covers the setting audited here is decided by three questions: whether the weights are overwritten in place onto the int8 grid with no high-precision master copy and the projection interval is swept rather than fixed; whether a shadow averages quantized weights and that average is written after projection; and whether the full int8 chain of a target NPU is measured and the intervals are reported across seeds. These six dimensions are mapped item by item in Table S1, two to a question, and no cited design answers all six. Text S1 groups the remaining hits of that search: [27], [28], [29], [30], [31] and [32], of which ProxQuant pulls continuous weights towards the grid with a proximal term [30] and BinaryRelax relaxes over a schedule of penalty strengths instead of projecting at all [33]. All of them change the update rule; this paper keeps the projection and changes only when it is applied.

## 3. Method

### 3.1 Model and task

The model is a small architecture (4.41 M parameters) whose last layers stay quantization-friendly:

* regnety_008 backbone (RegNetY, from the RegNet design space [34]) as implemented in timm [35], feature levels 3 and 4, ImageNet-pretrained, input 224×224 (UTKFace, released with [36]);
* a lightweight FPN (1×1 lateral projections to 256 channels, bilinear ×2 upsampling, one shared GroupNorm(8) [37], additive fusion) and AdaptiveAvgPool2d to a 14×14 grid (Text S2);
* a 2-layer TransformerEncoder (PyTorch `nn.TransformerEncoder`; 8 heads, Pre-LN, feed-forward 4d; architecture of [38]);
* a ReLU-QAT control head, Linear(256→256) → RMSNorm → ReLU → Linear(256→1), 66.3 k parameters; RMSNorm [39] avoids mean subtraction and ReLU avoids transcendental operators.

Task: UTKFace age regression, target = age/100 (a MAE of 0.01 is one year), L1 loss, 23,708 images, random 15 % validation split of 3,556 images.

### 3.2 The snap operator

For a weight matrix $W \in \mathbb{R}^{C_{out} \times C_{in}}$ quantized to $b$ bits, the integer range (two's complement, one extra step negative) and the per-output-channel scale are

$$q_b = 2^{b-1} - 1$$

$$s_o(W) = \frac{\max_j \left| W_{oj} \right|}{q_b}$$

and the projection is applied element-wise:

$$[S_b(W)]_{oj} = s_o(W)\cdot\mathrm{clip}\left(\mathrm{round}\left(\frac{W_{oj}}{s_o(W)}\right),\, -q_b-1,\, q_b\right)$$

The relative residual of one application over all projected tensors is

$$\varepsilon_b(W) = \frac{\left\| W - S_b(W) \right\|_2}{\left\| W \right\|_2}$$

and returned at every step. On a fixed grid the operator is a projection in the strict sense: the channel maximum maps to $\pm q_b$ and is preserved by the clip, so $S_b(S_b(W)) = S_b(W)$ holds to floating-point precision, and each application erases the sub-cell offset inside a cell rather than the row scale (§5.4). Drift between steps and the clip bound are in Text S2.

The operator overwrites the parameters the optimizer holds, so gradients accumulate on the quantized weights and there is no full-precision master copy; the difference from straight-through fake quantization is set out in Text S2.

### 3.3 Three update protocols

All three run with $\gamma = 0.99$ and update once per optimizer step (Table S4). They differ in what they export. The *live* protocol keeps no shadow and exports the last projected iterate, still projected every $N$ steps, since only $N = \infty$ switches projection off. The *ema_before* protocol averages iterates that have not yet been projected and projects once at export, and *ema_after* averages projected iterates, so the exported average follows the grid throughout. The released code still takes the old values (`none`, `before_snap`, `after_snap`) on the command line. With the EMA operator

$$E_\gamma(m, W) = \gamma m + (1-\gamma) W$$

at every optimizer step $t$ the three protocols compute, with $P_N$ the projection applied only on steps that are multiples of the interval,

$$P_N(W_t) = S_b(W_t)\ \ \text{if}\ \ t \equiv 0\ (\mathrm{mod}\ N),\ \ \ \ P_N(W_t) = W_t\ \ \text{otherwise}$$

$$m_t = E_\gamma(m_{t-1}, W_t),\ \ \ \ W_t \leftarrow P_N(W_t)$$

$$W_t \leftarrow P_N(W_t),\ \ \ \ m_t = E_\gamma(m_{t-1}, W_t)$$

$$W_t \leftarrow P_N(W_t)$$

The setting $N = 1$ projects at every step and $N = \infty$ never projects; the first two lines export the shadow $m_T$ and the third the live weights $W_T$. The two operations do not commute,

$$S_b(E_\gamma(m,W)) \neq E_\gamma(m, S_b(W))$$

which does not rest on the nonlinearity of $S_b$: $S_b$ acts on only one of the two terms, rounding the convex combination on the left and only $W$ on the right, and the two sides agree only under narrow conditions (Text S2). Figure 1 draws the three protocols and the straight-through control.

![Fig 1](figs/fig1.png)

**Figure 1.** Three in-place protocols and the straight-through estimator (STE) control. $S$ projects the weights onto a symmetric int8 grid every $N$ steps; $E$ updates the shadow; $N = \infty$ switches projection off.

### 3.4 Deployment measurement

Training-time projection and export-time quantization are measured separately: at export we take the deployed weights ($m_T$ or $W_T$) and apply a quantize-dequantize pass at $b$ bits with the same per-output-channel rule,

$$\hat{W} = S_b(W_{\mathrm{dep}}),\ \ \ W_{\mathrm{dep}} \in \{m_T, W_T\}$$

and evaluate the task metric on the validation split,

$$\mathrm{MAE} = \frac{1}{M}\sum_{i=1}^{M}{\left| \hat{y}_i - y_i \right|}$$

read next to the zero baseline, the constant predictor that outputs the validation-split mean:

$$\mathrm{MAE}_0 = \frac{1}{M}\sum_{i=1}^{M}{\left| \bar{y} - y_i \right|},\ \ \ \ \bar{y} = \frac{1}{M}\sum_{j=1}^{M}{y_j^{\mathrm{val}}}$$

Weight-space differences are relative L2 distances between flattened vectors of projected tensors,

$$d(W, W') = \frac{\left\| W - W' \right\|_2}{\left\| W \right\|_2}$$

$b = 8$ reproduces the target 8 bit deployment; $b = 6, 4$ are stress tests. The controlled sweeps of §5.1–§5.4 rest on this weight-only simulation; the vendor toolchain is run in §5.5.

Notation used throughout: $\Delta$ is the tested quantity minus its reference, so a positive value means the tested one is worse; against FP32 the tested quantity is the quantized model (Tables 2 and 4 and Tables S16–S19).

## 4. Experimental setup

The validation split is rebuilt per run (Table S3): the run's seed sets both the data split and the parameter initialization, and the two are not decoupled. All numbers here use each run's own split, and the FP32 column is checked against the training log's validation MAE (agreement within 0.005, the difference being the 400-image subsample). Two accounting bases stay apart: `history.csv` reports MAE on 2,000 images sampled at random from the 3,556-image validation split (`--val-windows` defaults to 2000), while the zero baseline 0.1553 is computed on all 3,556. The deployment table takes the first 400 images of the same split, whose own zero baseline is 0.1625 for seed 0 and 0.1489 and 0.1483 for seeds 1 and 2. To remove the split from cross-seed comparisons, the five main sequences share one fixed split (`split_seed` = 0) and vary only the initial seed (0–7), so their run-to-run spread contains initialization and batch-order differences only; the remaining cells keep a split per run and are read descriptively (Text S2).

The basis for run-level inference is five main-configuration sequences of 8 initial seeds each, 40 runs at 20 epochs; the run inventory, including older batches, is in Text S2. The deployment tables cover 65 of the 70 runs listed there, 50 of them on the 400 images shared with the main basis; 46 runs are also read on the 1,556-image external subset that carries the run-level contrasts (§5.2). Table 1 fixes the quantities on which every number in §5 rests.

Every run optimizes the same objective: AdamW at a learning rate of $10^{-4}$ under cosine annealing over 20 epochs, batch 64, bf16 autocast, gradient clipping at 1.0 and an EMA decay of 0.99, with the training-time projection at 8 bits (Table S3). The architecture axis of §5.5 reuses that recipe at $10^{-3}$ and adds a second, under-trained axis at $10^{-4}$; the classification arm fixes its own schedule in Table S24.

Statistically, three families plus one equivalence test are pre-specified: the interval effect (intervals 1 and 20, contrasts a and b), the presence of the shadow (c), STE against in-place overwriting and against projection off (two contrasts), and the equivalence test for contrast d. This yields six run-level contrasts in total (Table S14, collected in the Supplementary Material, Text S7). Every other comparison in §5 is exploratory and uncorrected for multiplicity (Text S2).

**Table 1.** The quantities on which every number in §5 rests: what is measured, on how many images, and which runs support it.

| Quantity | Value | Where it is documented | Read in |
|---|---|---|---|
| Training images | 23,708 UTKFace faces, 224 × 224, age taken from the file name | Text S2 | every run |
| Split | 15 % validation per run, split seed fixed at 0 in the main sequences; the epoch is chosen on a 2,000-image monitoring subset | Text S2 | all training and selection |
| Main sequences | five configurations × 8 initial seeds = 40 runs at 20 epochs | Text S2, Table S13 | run-level inference (§5.2-§5.5) |
| Deployment subset | 400 images, overlapping the monitoring subset | Table S13 | descriptive only (§5.1, §5.3) |
| Held-out basis | 1,556 images that took no part in choosing an epoch | Table S13 | pre-specified contrasts (§5.2-§5.5) |
| Deployment coverage | 65 of the 70 runs of Text S2; 46 also read on the held-out basis | Text S2 | intervals of Table 3 |
| Classification arm | CIFAR-10, 50,000 training and 10,000 test images, ResNet-18 with a 3 × 3 stem, six arms of eight seeds at 60 epochs | Text S12, Table S24 | §5.5 |

Table S14 fixes the statistical identity of every claim: the inference unit and its sample size, the number of contrasts, whether the comparison was pre-specified before the run-level results were inspected, and whether a multiple-comparison correction was applied. Every subsection of §5 has a row there; wherever "pre-specified" reads "no", the reading is exploratory only.

The analysis was pre-planned but never externally registered, and we use the word in that limited sense: the six pre-specified contrasts, their directions, the 1,556-image basis, the 8 bit primary width and the 0.0016 bound were fixed before the run-level readings on the external subset were computed. What changed afterwards was the run inventory rather than the hypotheses: the five main sequences and the STE arm were topped up from three seeds to eight, and a third deployment backend was run, all entering contrasts already defined. The plan and every departure from it are frozen in `preregistration_freeze.md` inside `quant_runs/`; the released repository is a single squashed commit, so the file is identified by its SHA-256 in `manifest.json` rather than by a commit hash, and its content cannot be independently dated, so what it fixes is the plan's content rather than its precedence; the same contrasts on the first three seeds and on the delivered eight are tabulated in Text S8 (Table S20), with no sign flips and a largest change of 0.001125 on the exploratory learning-rate row.

The estimand is the mean paired difference over re-initializations at this split and budget, not an effect generalized over data splits. Table S10 separates the two sources in a $3\times3$ cross of split and initialization: the split accounts for 56.6 % of the sum of squares, initialization for 10.0 %, and the residual, which includes their interaction, for 33.4 %, with neither factor significant at four residual degrees of freedom. The interval ordering survives a change of architecture and dataset (§5.5), but the absolute level is conditional on the split, and the eight seeds quantify initialization noise rather than split noise.

The six pre-specified contrasts form one family because they are computed on the same 1,556-image external subset and the same eight paired seeds; Holm adjustment therefore controls the family-wise error rate at $\alpha = 0.05$ across the six, including the equivalence test for contrast d. Three distribution-free readings of the same eight pairs — an exact Wilcoxon signed-rank test, a percentile bootstrap interval and the standardised paired effect size — are in Text S10 and Table S22; none of them switches a decision.

## 5. Results

### 5.1 At 8 bits, the export quantizer is a no-op

At interval 1 the live protocol's exported weights sit on the grid; at longer intervals the optimizer steps after the last projection, so they are off it at export (relative RMS residual 0.0037 at interval 20), and averaging pushes the exported shadows off it (Fig. S2). Re-quantizing either kind leaves the metric unchanged at four decimals. The first measurement is the residual against the channel grid, in grid steps,

$$\rho_{oj} = \frac{\left| W_{oj} - [S_b(W)]_{oj} \right|}{s_o(W)}$$

the second, the fraction of weights indistinguishable from a grid point,

$$f_{\tau_g}(W) = \frac{1}{\left| W \right|}\sum_{oj}{\mathbf{1}\left[ \rho_{oj} < \tau_g \right]},\ \ \ \ \tau_g = 0.05\ \text{grid steps}$$

The residual column of Table S5 is the relative RMS of eq. (4), a norm over the weight vector, not the mean of $\rho_{oj}$.

The raw cosine is dominated by the vector mean, so the informative form is the centered cosine, the Pearson correlation of the two flattened vectors ($\mu_W$ and $\mu_{\hat W}$ are means over all entries):

$$\cos_c(W, \hat{W}) = \frac{\sum_{o,j}{\left(W_{oj} - \mu_W\right)\left(\hat{W}_{oj} - \mu_{\hat{W}}\right)}}{\sqrt{\sum_{o,j}{\left(W_{oj} - \mu_W\right)^2}}\ \sqrt{\sum_{o,j}{\left(\hat{W}_{oj} - \mu_{\hat{W}}\right)^2}}}$$

It equals 1 whenever $\hat{W} = W$, so an exact identity tells a report nothing; what a report needs is the resolution, how far from 1 the true value can sit while the printed value stays at 1.0000.

```latex
\begin{proposition}[Resolution bound on the export fidelity metric]
\label{prop:noinfo}
Let $\hat W = S^{\mathrm{dep}}_b(W)$ be the exported tensor after the deployment quantizer,
$\varepsilon = \lVert\hat W - W\rVert_2/\lVert W\rVert_2$ the relative residual of eq.~(4), and
$\kappa = \lVert W - \mu_W\rVert_2/\lVert W\rVert_2$ the shortening that centering applies to $W$.
Then the centered cosine of eq.~(17) obeys
\[
\cos_c(W,\hat W) \;\ge\; 1 - 2\left(\varepsilon/\kappa\right)^2 .
\]
If the cosine is quoted to $d$ decimals, with step $h = 10^{-d}$, and $2(\varepsilon/\kappa)^2 \le h/2$,
then $\cos_c \in [1-h/2, 1]$ for every such checkpoint, so the printed value is $1.0000$ whatever its
task accuracy and no two checkpoints can be ranked by it. The bound is per tensor; the reported cosines average over projected tensors, so the test is applied tensor by tensor, and a mean of $1.0000$ follows only if every tensor saturates. When the four conditions of the identity case
hold, $\varepsilon = 0$ and the bound is attained.
\end{proposition}
\begin{proof}
Write $W_c = W - \mu_W\mathbf{1}$ and $\hat W_c = \hat W - \mu_{\hat W}\mathbf{1}$, so that
$\hat W_c = W_c + e$ with $e = (\hat W - W) - (\mu_{\hat W} - \mu_W)\mathbf{1}$. Subtracting the mean
is an orthogonal projection onto the zero-mean subspace, hence non-expansive:
$\lVert e\rVert_2 \le \lVert\hat W - W\rVert_2 = \varepsilon\lVert W\rVert_2$. For nonzero $u$ and $v$,
$\lVert u/\lVert u\rVert - v/\lVert v\rVert\rVert \le 2\lVert u-v\rVert/\lVert u\rVert$. With $u = W_c$,
$v = \hat W_c$ and $\lVert W_c\rVert = \kappa\lVert W\rVert$ this gives $\lVert\hat u - \hat v\rVert \le 2\varepsilon/\kappa$,
and $\cos_c = 1 - \frac{1}{2}\lVert\hat u - \hat v\rVert^2$ yields the bound. A residual of zero makes $e$
zero, so $\hat u = \hat v$ and the cosine is exactly 1.
\end{proof}
```

Both quantities are measurable on a checkpoint, and `scripts/resolution_bound.py` reports them beside the bound (`quant_runs/resolution_bound.json`). Centering shortens the exported tensors by 0.01 %, so $\kappa = 0.9999$ on all six checkpoints, and at four decimals the bound tolerates $\varepsilon \le \kappa\sqrt{h/4} = \kappa\sqrt{h}/2 = 5.0\times10^{-3}$. The three protocols sit inside that limit — across sixteen shadow checkpoints at interval 20 the residual runs from 2.9e-3 to 4.1e-3, and the live protocols are at 8e-9 and 3.5e-5 — so their printed cosine is fixed at 1.0000 and cannot order them. The endpoint that switches projection off is outside it ($\varepsilon = 1.0\times10^{-2}$) and its cosine does move, printing 0.9999; a 4 bit export is further out ($1.8\times10^{-1}$) and reads 0.985. Table S7 reports the residual per protocol under both scale provenances, and the bound is slack at every row.

Two distances have to be kept apart: the one the export projection produces, between the tensor it is given and the deployed weights (8e-9 to 4.1e-3), which the cosine sees, and the task-level difference between the FP32 and the 8 bit model on the same images (0.0000–0.0014, Table S5), which it does not. The 6 bit and 4 bit readings below are post-training quantization at export with the training-time projection fixed at 8 bits (Table S3). Why the live weights at interval 1 export at 8e-9 is structural: a projection preserves the per-row maximum, so the export scale equals that of the last training step and the export moves nothing. What makes the metric uninformative is not that the weights are identical but that these distances sit one to two orders of magnitude below the differences between protocols (Table S6).

```latex
\begin{algorithm}[!t]
\caption{Admissibility pre-check before a deployment fidelity metric is quoted}
\label{alg:precheck}
\begin{algorithmic}[1]
\Require training rule $S_b$, deployment quantizer $S^{\mathrm{dep}}_b$, exported $W_{\mathrm{dep}}$, quoted decimals $d$ (step $h = 10^{-d}$), tolerance $\tau_R$ (default $\tau_R = \kappa\sqrt{h}/2$)
\Ensure an admissible metric, or the reason to drop one
\State $R \gets \lVert W_{\mathrm{dep}}-S^{\mathrm{dep}}_b(W_{\mathrm{dep}})\rVert_2 / \lVert W_{\mathrm{dep}}\rVert_2$
\State $C \gets W_{\mathrm{dep}} - \mu\mathbf{1}$; \quad $\kappa \gets \lVert C\rVert_2/\lVert W_{\mathrm{dep}}\rVert_2$
\If{$\lVert C\rVert_2/\lVert W_{\mathrm{dep}}\rVert_2 < 10^{-3}$}
  \State \Return the centered cosine is undefined or the bound is vacuous (the centered tensor is near zero); report task MAE instead
\EndIf
\If{operator, step or rounding differ between $S_b$ and $S^{\mathrm{dep}}_b$}
  \State \Return $R$ under the deployment step, eq.~(15); a cosine is not comparable
\EndIf
\If{$2(R/\kappa)^2 \le h/2$, i.e. $R \le \tau_R$}
  \State \Return drop the fidelity metric (Proposition~\ref{prop:noinfo}); report task MAE after export quantization
\EndIf
\State \Return the centered cosine, with its interval, bit width and evaluation basis
\end{algorithmic}
\end{algorithm}
```

Rows whose exported maximum is zero (an all-zero output channel) carry no scale; the deployment rule leaves them unchanged, they add nothing to $R$, and they are excluded from the cosine, which is then taken over the remaining entries.

The tolerance $\tau_R$ is the one free number in the check, and the bound sets its scale: at four decimals the proposition tolerates $\varepsilon \le \kappa\sqrt{h/4} = \kappa\sqrt{h}/2$, which is $5.0\times10^{-3}$ at $\kappa = 0.9999$. We set $\tau_R = 5\times10^{-3}$, the bound itself for $\kappa \approx 1$ and 1.2 times the largest residual our own chains produce ($4.1\times10^{-3}$, Table S7), so the check never calls a metric uninformative unless the bound proves it; a checkpoint between the two numbers would be quoted with provenance although its cosine is probably stuck as well, and none of our protocols falls in that gap. The projection-off endpoint exports a tensor no training step has touched, so its $1.0\times10^{-2}$ falls through to the last line and its cosine is quoted only with provenance. The check runs as a single entry point on a checkpoint directory, and Table S13 maps it.

The no-op concerns the weight quantizer only: re-projecting projected weights changes nothing (§3.2), whereas a deployment compiler also quantizes *activations* into look-up tables, and that part is not idempotent. The full pipeline (§5.5) costs $+0.0034$–$0.0042$ MAE across the ReLU-QAT and GLU heads, an order of magnitude more than the 0.0004 of Table S5, and it leaves the interval ranking intact.


### 5.2 The cost of projection, and the endpoint that switches it off

Task MAE after deployment quantization, seed 0, three protocols per interval, plus the endpoint N = ∞ at which projection is switched off (Table 2):

**Table 2.** Task MAE after deployment quantization (seed 0, three intervals and the projection-off endpoint) on the 400-image subset, with the 1,556-image external subset in the last column.
| interval | protocol | FP32 | 8 bit | 6 bit | 4 bit | Held-out FP32 |
|---|---|---|---|---|---|---|
| 1 | live | 0.0741 | 0.0741 | 0.0785 | 0.0900 | 0.0741 |
| 1 | ema_before | 0.0741 | 0.0741 | 0.0776 | 0.0899 | 0.0738 |
| 1 | ema_after | 0.0748 | 0.0748 | 0.0794 | 0.0910 | 0.0738 |
| 5 | live | 0.0660 | 0.0662 | 0.0672 | 0.0857 | 0.0652 |
| 5 | ema_before | 0.0658 | 0.0658 | 0.0674 | 0.0880 | 0.0653 |
| 5 | ema_after | 0.0657 | 0.0657 | 0.0671 | 0.0864 | 0.0653 |
| 20 | live | 0.0619 | 0.0620 | 0.0642 | 0.0888 | 0.0591 |
| 20 | ema_before | 0.0607 | 0.0608 | 0.0643 | 0.0866 | 0.0579 |
| 20 | ema_after | 0.0594 | 0.0593 | 0.0631 | 0.0849 | 0.0573 |
| ∞ | live | 0.0552 | 0.0548 | 0.0575 | 0.1535 | 0.0543 |
| ∞ | ema_after | 0.0556 | 0.0554 | 0.0569 | 0.1040 | 0.0546 |
**Note:** Point estimates only; the paired differences, their 95 % confidence intervals and the 1,556-image basis are in Text S3.2 and quant_runs/paired_report.md.

The two $N = \infty$ rows mean no projection during training (`snap_every` = 0). ema_before and ema_after then collapse into the same algorithm, so only ema_after is listed, and live exports its weights directly. Their FP32 values, 0.0556 and 0.0552, differ by +0.00043 with an interval that contains zero (§5.3).

![Fig 2](figs/fig2.png)

**Figure 2.** (a) Deployment MAE against projection interval; $N = \infty$ is projection off, the band is the range of the two shadow protocols wherever they differ, and they coincide at $N = \infty$. (b) 4 bit spread over eight seeds.

At the run level the drop becomes two contrasts against switching projection off. With eight
paired seeds on one fixed split (df = 7) and scored on the 1,556 images that took no part in
epoch selection, interval 1 reads $+0.0185$ and interval 20 $+0.0047$; neither interval
contains zero (Table 3). At 8 bit, projection is a *cost* paid in optimization noise. The
longer the interval, the smaller that cost, and both contrasts sit far above the run-to-run
dispersion (paired sd 0.0011 and 0.0008).

**Table 3.** Run-level contrasts: $\Delta$ is the tested quantity minus its reference in deployment MAE on the 1,556 images held out of epoch selection, brackets are 95 % paired $t$ intervals over eight seeds (df = 7). The Holm values of the first block cover those six pre-specified contrasts; the 6 bit block is a post hoc family of four, and the controls belong to neither family. The setting names the tensor whose deployment MAE is scored. FP32 is the run's exported weights evaluated as they are, the basis of every interval quoted in the text; 8 bit is the same weights after export re-quantization, which is where stochastic rounding and the straight-through control are read. Off abbreviates projection switched off.
| Contrast | Setting | $\Delta$ MAE | 95 % CI | Same sign | $p$ | Holm |
|---|---|---|---|---|---|---|
| a: interval 1 $-$ off | FP32 | $+0.0185$ | $[+0.0175, +0.0194]$ | 8/8 | $6.4\times10^{-10}$ | $3.9\times10^{-9}$ |
| b: interval 20 $-$ off | FP32 | $+0.0047$ | $[+0.0040, +0.0054]$ | 8/8 | $8.8\times10^{-7}$ | $3.5\times10^{-6}$ |
| c: shadow at interval 20 | FP32 | $-0.0014$ | $[-0.0018, -0.0010]$ | 8/8 | $1.1\times10^{-4}$ | $2.2\times10^{-4}$ |
| d: order at interval 20 | FP32 | $+0.00003$ | $[-0.00025, +0.00031]$ | 5/8 | $1.5\times10^{-6}$ (TOST) | $4.6\times10^{-6}$ |
| STE minus in-place overwrite | 8 bit | $-0.0184$ | $[-0.0196, -0.0173]$ | 8/8 | $2.1\times10^{-9}$ | $1.0\times10^{-8}$ |
| STE minus projection off | 8 bit | $-0.0001$ | $[-0.0008, +0.0006]$ | 4/8 negative | $0.76$ | $0.76$ |
| a: interval 1 $-$ off | 6 bit | $+0.0214$ | $[+0.0195, +0.0234]$ | 8/8 | $3.3\times10^{-8}$ | $1.3\times10^{-7}$ |
| b: interval 20 $-$ off | 6 bit | $+0.0060$ | $[+0.0042, +0.0079]$ | 8/8 | $9.8\times10^{-5}$ | $2.9\times10^{-4}$ |
| c: shadow at interval 20 | 6 bit | $-0.0027$ | $[-0.0044, -0.0009]$ | 7/8 | $0.0092$ | $0.018$ |
| d: order at interval 20 | 6 bit | $-0.00029$ | $[-0.00109, +0.00051]$ | 4/8 | $0.42$ | $0.42$ |
| a at 60 epochs | FP32 | $+0.0138$ | $[+0.0127, +0.0149]$ | 8/8 | $1.1\times10^{-8}$ | - |
| a at 30-fold learning rate | FP32 | $+0.0055$ | $[-0.0009, +0.0118]$ | 6/8 | $0.08$ | - |
| change in a: 60 vs 20 ep | FP32 | $-0.0047$ | $[-0.0057, -0.0036]$ | 8/8 negative | - | - |
| change in a: lr 30-fold | FP32 | $-0.0130$ | $[-0.0192, -0.0068]$ | 7/8 negative | - | - |
| a: stochastic rounding | 8 bit | $+0.00004$ | $[-0.0006, +0.0007]$ | 5/8 negative | $0.89$ | - |

The endpoint gives the lower end of the cost curve: with projection off the FP32 deployment MAE is 0.0552 (live) and 0.0556 (ema_after), 0.0189 and 0.0192 below interval 1 and about 0.0067 and 0.0037 below interval 20 (single-run point estimates, Table 2). At $N = \infty$ the 8 bit minus FP32 difference is still only 0.0002–0.0004, written −0.00037 and −0.00023 under the convention of §3.4 with both intervals containing zero, and most finite-interval runs hold it within 0.0004 (0.0000 at interval 1), the two additional runs aside.

Hence at the weight-only 8 bit target training-time projection is a pure cost and the optimal setting is to switch it off; at 6 bits the same contrasts are larger, $+0.0214$ and $+0.0060$, 8/8 of one sign (Table 3). The cost is consistent with the in-place overwrite rather than with weights resting on the grid, and on the 1,556-image basis the STE control reads $-0.0001$ against switching projection off (§5.4). The three-seed, 400-image reading of 0.0008 is in Table S11 for completeness and is not the inferential basis.

The evaluations above use the first 400 images of each run's validation split, which overlaps the 2,000-image monitoring subset the best epoch was selected on (§4); re-running the same quantizer on the 1,556 images that took no part in selection removes the overlap (zero filename overlap, predictions archived), and Fig. S3 gives the per-seed paired differences under both bases. There the interval effect is unchanged (0.074110 at interval 1 against 0.054271 with projection off, $+0.01984$ per image, se 0.00138, 95 % CI $[+0.01713, +0.02255]$, as in the 0.0189 of Table 2). The ordering effect halves on that run: $+0.00132$ on the 400 images against $+0.00066$ on the same run's 1,556 images (se 0.00038, CI $[-0.00008, +0.00139]$, still within this test's resolution), against a run-level mean of $+0.00003$ over the eight seeds (§5.3). The selection optimism is measurable: the 400-image FP32 MAE exceeds the external subset by 0.0012 on average, growing with the interval ($+0.00044$ at 1, $+0.00055$ at 5, $+0.00256$ at 20), while the shadow survives ($-0.00118$ and $-0.00184$ at interval 20, neither interval containing zero) and disappears with projection off ($+0.00035$). Epoch selection and the STE control: ESM, Text S5.

### 5.3 What the shadow and the averaging order change

Within one interval at seed 0 the three protocols span 0.0027 (8 bit, interval 20) and 0.0025 (FP32, interval 20), of which the ordering accounts for 0.0015 and 0.0013; at 6 bits the run-level ordering contrast is $-0.00029$, 4 of 8 positive (Table 3), and the largest single-seed gap is 0.0018 at interval 1.

These are single-run descriptions, so the inferential unit becomes the run: five main sequences (live at $N = 1$, $N = 20$ and $N = \infty$, plus ema_before and ema_after at $N = 20$) carry 8 initial seeds on one fixed split, paired run by run. Contrast d (ema_before $-$ ema_after at $N = 20$) reads $+0.00003$ on the 1,556 images that never took part in epoch selection, 5 of 8 positive, and $+0.00018$ on the 400-image subset; the per-seed paired sds of the four run-level contrasts are 0.0011, 0.0008, 0.0005 and 0.0003.

The equivalence margin is pre-specified and independent of the observations: 1 % of the 0.1553 zero baseline, i.e. 0.0015526728, or 0.0016 to four decimals — a maximum negligible degradation budget of 0.16 years, an order of magnitude below the 0.0183 that the projection costs at interval 1. A two one-sided test (TOST) [40], with the tutorial in [41], rules out an ordering effect of 0.0016 or more on the 1,556 images (raw $p = 1.5\times10^{-6}$, Holm-adjusted $4.6\times10^{-6}$) and on the 400 ($p = 0.00064$); the minimum detectable effect at 80 % power is 0.000383 and 0.000897. A post-hoc sensitivity analysis puts the critical bound at 0.000252 and returns $p$ between $1.2\times10^{-8}$ and 0.027 over bounds from 0.0003 to 0.0032, so every bound leads to the same conclusion (Text S3.3). The bound was not set so that the test would pass.

The 6 bit contrast d is reported for completeness only: its paired standard error is 0.000338, 2.9 times the 0.000118 of the 8 bit basis, so the 80 % MDE rises to 0.0011, 4 of the 8 seeds fall on each side, and the pre-specified bound is only 1.4 times that MDE. No equivalence claim is attached to 6 bits. The other three 6 bit contrasts keep their direction ($+0.0214$, $+0.0060$, $-0.0027$ with 7/8 of one sign) and, taken as one post-hoc family of four and corrected by Holm, give $1.3\times10^{-7}$, $2.9\times10^{-4}$, $0.018$ and $0.42$, so these directions do not rest on an uncorrected multiplicity choice and nothing is promoted to pre-specified. Every sign test quoted beside these intervals is bounded below by $p = 0.0078$ at $n = 8$.

The seven per-image paired comparisons made earlier are exploratory only: none exceeds 0.0014, the sign changes with the seed, and 0/7 survive Holm and Benjamini–Hochberg correction (Text S3.4). The six pre-specified contrasts also form one family: on the 1,556-image basis the paired $t$ statistics give $p = 6.4\times10^{-10}$ for a, $8.8\times10^{-7}$ for b, $1.1\times10^{-4}$ for c, $2.1\times10^{-9}$ for STE against in-place overwriting and $0.76$ for STE against projection off, Holm leaves every conclusion unchanged, and the equivalence test enters the same family at $4.6\times10^{-6}$.

The ordering is not a no-op on the weights: the exported weight vectors differ by 4.3–9.1 % in relative L2 (Table S6), yet the task metric moves by $+0.000029$ (1,556-image basis, interval containing zero) where retraining with another seed moves 64–83 % (Fig. S6). A claim that one ordering is "safer" must be argued in function space, and this measurement does not support it. Export quantization does not produce the difference either: requantizing both orders at interval 20 moves the relative L2 from 4.7 % ± 0.6 % to 4.8 % ± 0.6 %, after which 98 % of the weights still land on different grid points, and the cross-seed share is 63.5 % before and after (8 seeds, fixed split; artifact in Table S13). That 4.8 % is not $\rho$ of §5.1: it is the relative L2 between two ordered weight sets, whereas $\rho$ is the residual of the deployed weights to their grid ($\varepsilon$ of eq. 4, at most 0.0037 at interval 20; Table S5).

Nor is the averaging time constant a control: with $\kappa = N(1-\gamma)$, at interval 20 and seed 0, $\gamma$ = 0.9, 0.99 and 0.999 give $\kappa$ = 2, 0.2 and 0.02, and the ordering differences read +0.0014, +0.0013 and 0.0000 — a span that does not grow monotonically with $\kappa$ and, with the cross-seed results above, is not stable in sign.

The shadow does leave a small, consistently signed reading. At interval 20 the paired difference of ema_after against live is $-0.0014$ on the 1,556 held-out images, 8/8 runs of one sign (Table 3); on the 400-image subset it is $-0.0012$, 7/8 ($p = 0.070$), a subset that overlaps the 2,000-image monitoring set and so deflates run-to-run dispersion. The effect exists at 0.0014, an order of magnitude below the interval effect, and it vanishes in the protocol mean: at interval 20 and seed 0 the protocols read 0.0619, 0.0607 and 0.0594, and the mean coincides with ema_before. Nor is it unconditionally useful: with projection off, ema_after against live reads $+0.00043$ at the image level with an interval containing zero. Unlike SWA [15] or model soups [16] the shadow needs no retraining and slides along a single trajectory, which is why its magnitude is 0.0014. At 6 bits the contrast is twice as large ($-0.0027$) but only 7 of 8 runs share its sign, so we report the interval without treating it as stronger evidence.

### 5.4 Why the cost appears: the overwrite, not the grid

Three diagnostics — the per-event projection error, the share of projections that were fixed points, and the shadow-to-live distance — are in Text S4; none pins the model to the grid.

Projection rewrites the parameters themselves; fake quantization does not. A standard straight-through estimator (STE) puts the quantization in the forward pass, $w_q = w + (\mathrm{snap}(w) - w).\mathrm{detach()}$, so the parameters stay float: the weights count as quantized at every step, yet no step overwrites them. Eight initial seeds ran on the fixed split with three projection modes — STE, in-place overwrite at interval 1, and projection off — every other hyperparameter identical (Table S11). Paired seed by seed on the 1,556 held-out images under 8 bit, STE sits 0.0184 below the overwrite (8/8), the overwrite 0.0183 above projection off (8/8), and STE against projection off reads $-0.0001$ (4 of 8 negative; Table 3). The group means are 0.0548 ± 0.0012, 0.0732 ± 0.0011 and 0.0549 ± 0.0007: lying on the grid at every step costs nothing, replacing the parameters with grid points costs two orders of magnitude more.

The cost co-varies with the number of overwrite events (6,280 worst, then 1,260, 320, zero best), while the interval also sets the single-step displacement (relative RMS 0.000 → 0.0012 → 0.0030) in the opposite direction, so the two terms are measured separately in the next paragraph rather than inferred from the co-variation. Three mechanism controls set its scale and place it on the rounding rule and on how much progress each overwrite destroys. Tripling the budget to 60 epochs lowers the interval-1 cost from $+0.0185$ to $+0.0138$, 8/8, so slower convergence accounts for a quarter of it and a longer budget does not remove it. Raising the learning rate 30-fold lowers it to $+0.0055$, which eight seeds cannot separate from zero (6/8 positive), yet the contraction is resolved against the same runs at $10^{-4}$ ($-0.0130$, 7/8 negative). At 8 bits, replacing only the rounding at each projection with stochastic leaves $+0.00004$ against projection off (5 negative, 3 positive) where nearest rounding reads $+0.0183$; with a per-seed recovery of 94.4 % to 105.9 % (mean 99.8 %, Fig. 3), the fix removes 99.8 % of that cost on average at no change to grid, interval or budget. The per-control intervals and sign tests are in Table 3 and Text S4.

**What one overwrite costs.** Nine logged runs (three seeds at intervals 1, 5 and 20; 6,280, 1,260 and 320 events) give $u$, the drift since the previous projection, and $s$, the displacement the write applies. Their ratio is 0.39 at interval 1, 3.97 at interval 5 and 12.7 at interval 20, where the write moves the tensor further than the twenty steps before it (Text S11, Table S23). The drift between projections stays near two grid steps at every interval, so the tensor is reset after roughly the same travel. Per projection the cost rises 4.8-fold from interval 1 to 20 while the count falls 19.6-fold; on these seeds the axis falls 4.05-fold, bracketed by a frequency reading that under-predicts that fall and an erasure-times-frequency reading that over-predicts it (Text S11 gives both factors, with costs 0.0187, 0.0110 and 0.0046). Both readings have one free parameter, fitted on one interval and checked on two: order-of-magnitude probes, not a model comparison.

**Table 4.** Three ways of handling weights in training, their prior-work counterparts, and the readings here

| Handling during training | Counterpart | What that line changes | Reading under this recipe | Relation to the interval $N$ |
|---|---|---|---|---|
| Fake-quantize forward, parameters not overwritten (STE) | the straight-through estimator of standard QAT [4, 28, 42] | forward quantized on the grid, straight-through backward, parameters float | -0.0001 [-0.0008, +0.0006] against projection off, 4/8 negative | never triggers an overwrite, so independent of $N$: forward quantized but never projected |
| Overwrite in place every step, nearest rounding | direct quantized training without a master copy, deterministically rounded (control case of [5]) | writes the parameters to the dequantized grid value every step | +0.0183 [+0.0176, +0.0191] against projection off, 8/8 | this is the $N = 1$ cell of this paper |
| Overwrite in place every step, stochastic rounding | DQT's stochastic rounding [5], ECO's master-copy-free training [6], analyses of stochastic rounding [43] | changes only the rounding rule; grid, interval and budget unchanged | +0.00004 [-0.0006, +0.0007] against projection off, 5 negative and 3 positive | exactly the $N = 1$ recipe, differing only in the rounding rule |
| Interval $N$ (the variable here) | when to project inside a weight-transfer schedule, e.g. [10] | how many optimizer steps separate two projections | $N{=}1$ +0.0183, $N{=}20$ +0.0047, $N{=}\infty$ 0 | the variable of this paper; its relation to schedules is discussed below |

![Fig 3](figs/fig3.png)

**Figure 3.** (a) Per-seed deployment MAE under stochastic and nearest rounding, 8 bits; the band is one standard deviation of the projection-off seeds. (b) Share of the cost recovered, mean 99.8 %.

**Projection schedules and the interval $N$.** Work such as [10] schedules how fast the weights move between grid cells while keeping full-precision latent weights; our $N$ only fixes how many optimizer steps separate two full projections and is constant within a run, so the two objects differ and their numbers are not comparable (Table S12). A schedule that moves the weights only part of the way to a cell writes points between cells; then $S_b$ acts on one side of a convex combination only, and averaging and projection do not commute (eq. (10)). A sparse-then-dense schedule is likewise not a linear interpolation between two fixed-$N$ readings; the longer argument is in Text S4.

With the shadow averaging the forward fake-quantized weight instead of the float parameter, the deployed FP32 MAE over three seeds is 0.0555 ± 0.0022 against 0.0560 ± 0.0035 for the plain STE control: a difference of 0.0005, far below the seed spread.

### 5.5 External validity: bit width, the target NPU, architecture, dataset, task type, backends and subject subgroups

Six checks ask whether the results above survive a change of bit width, implementation chain, network, dataset, task type and deployment backend. Each is scored on its own basis and read as directional unless it carries eight paired seeds and a run-level interval; what they measure is the sign of the endpoint contrast, not its size, and the detail sits in Text S4. A subgroup breakdown (Table 5) and the attribution control on the classification arm sit beside them.

**Bit width.** At 4 bits the two bases disagree in sign: $-0.0635$ on the 21-run basis and $+0.0710$ on the eight-seed fixed split, where the within-interval seed sd (0.0476–0.0626) is 5–7 times the between-interval spread, so only the per-seed direction is readable. The interval-1 contrast is $+0.0710$ $[+0.0013, +0.1406]$, 2 of 8 seeds reversed (sign test $p = 0.289$), and the paired sd of 0.0833 exceeds the effect itself; the earlier reading that a longer interval buys robustness compared a lucky seed with an unlucky one, so we report 4 bits as non-decidable under this design. Fig. S5 and Table S9 use the 21-run basis (seeds 0–2, own split), which ranks interval 1 lowest (0.1224) while the primary basis ranks the opposite way (0.2070 at interval 1, 0.1484 at 20, 0.1361 off). Topping up to eight seeds widens the 4 bit sd to 0.0365–0.0774 against 0.0020 at 8 bit, so this is an observation about 21 runs, not about 4 bit quantization in general. At 8 and 6 bits interval 20 is the best finite interval and $N = \infty$ is best overall (Table 2). With projection off the two 4 bit rows separate by 0.05 (0.1040 with the shadow, 0.1535 for the live weights), but both are seed-0 single-run readings and cannot rank the protocols (Text S4).

**The target NPU chain.** Everything above isolates one factor by simulating the quantizer in PyTorch, so the vendor toolchain was also run end to end on five checkpoints (four control-head exports plus one projection-off). Each goes through the same chain: export to a static ONNX graph, calibrate on 100 training-split images disjoint from the test subset, compile for the target BPU, and run the instruction stream on the vendor's x86 instruction-level simulator (Fig. S7). The compiled pipeline adds per-tensor percentile quantization of activations and lowers unary elementwise operations to `b30.lut`; over 200 test images paired image by image (single run), weight-only quantization costs at most 0.0003 and the projection-off checkpoint $+0.0040$ $[-0.0012, +0.0093]$. Provenance was established by re-running every checkpoint on a fixed input and matching it against the ONNX output ($\max|\Delta| \le 4\times10^{-7}$). The per-head detail — the output bias each deployment group introduces, the per-node cosines, and the eight-seed paired deltas of $+0.002916$ $[+0.001698, +0.004134]$ at $N = 1$ and $+0.001166$ $[-0.000400, +0.002732]$ at $N = \infty$ — is in Text S4 and Table S15.

On the target device the compiled graphs run at 1.13–1.20 ms per frame natively (820–878 FPS) and 1.50–1.54 ms through the Python path, with head-to-head differences below the measurement noise (Text S9). Two findings survive that detail. First, the compiled int8 model costs $+0.0034$–$0.0042$ MAE, an order of magnitude above the 0.0004 of §5.1, so the deployment grid is a larger error source than the projection (§8 item 7) and the pipeline is far from a no-op (Fig. S8). Second, switching the projection off removes only the cost the projection imposes, not the cost the chain imposes: $N = \infty$ is better than $N = 20$ by 0.0068 in FP32 and by 0.0065 in the compiled stream, i.e. 95 % of the training-side gain. On an RDK S100P (HBDK runtime 4.7.5, one BPU core) the board reproduces the simulator sample for sample over 800 inferences, averages 1.13–1.20 ms per frame, and the compiler's static latency estimate sits 12–18 % below it; Table S21 lists the per-model medians of the Python path.

**Architecture.** ResNet-18 (11.18 M) was trained from ImageNet initialization at $10^{-3}$ and paired seed by seed with the slim net over eight initial seeds on one fixed split, under the same projection rule and int8 recipe. Pooled over the two protocols of that axis, projection off reads 0.0507 and interval-1 overwriting 0.0580 on the 400 validation images of that axis, a difference of $+0.0072$ (1.14 times) that keeps the ordering of Table 2; on the 1,556 images that took no part in epoch selection the same pooling reads 0.0492 and 0.0563 ($+0.0071$), so the size is not an artefact of the basis. The 8 bit export adds at most 0.0004 in any cell, and at 6 and 4 bits the pooled difference grows to $+0.0092$ and $+0.0424$ with the same sign. A second axis at $10^{-4}$ leaves the network far from converged (0.1290 at interval 1 against 0.0564 with projection off, both under live), so those cells measure how much progress each projection destroys while the budget is short, not the cost at convergence: against projection off at that learning rate, intervals 1, 5 and 20 read $+0.0726$ $[+0.0585, +0.0866]$, $+0.0432$ $[+0.0386, +0.0479]$ and $+0.0138$ $[+0.0130, +0.0146]$, each 8/8 positive. The shadow reads $-0.0002$ $[-0.0010, +0.0005]$ (2/8) and the ordering $[-0.0005, +0.0004]$ (5/8), an interval of width 0.0009 against 0.0006 in the main setting and with a bound borrowed from it.

**Dataset.** KonIQ-10k [44] with a 1,510-image validation split is a different task, input size and budget, read on MAE against a zero baseline of 0.0044. Projection off reads 0.0020 and interval-1 overwriting 0.0027 on 400 validation images of that split, a difference of $+0.0007$ (1.36 times; 16 % of that baseline) in the same direction, and on the 400 images disjoint from the monitoring set it is again $+0.0007$ (0.00264 against 0.00195); the 8 bit export adds $+0.0007$, 6 bits $+0.0005$, and the shadow reads 7/8 positive ($p = 0.070$) for live and 8/8 ($p = 0.0078$) for ema_after. One run per cell; the split is tied to the seed on this axis, so within a seed the two arms share the split, the initialization and the evaluation images and their difference is paired, whereas across seeds the evaluation images differ and an interval would also carry split variance, which is why none is quoted; at 4 bits the two axes give opposite signs: a direction only.

**Task type.** A classification arm asks whether the cost belongs to the regression loss or to the overwrite. Six arms of eight seeds were trained on CIFAR-10 (50,000 training and 10,000 test images, ResNet-18 with a 3 × 3 stem, 60 epochs) under the same int8 rule. After export re-quantization, projection off reads 89.89 % top-1 and interval 1 reads 84.86 %, a cost of $+5.03$ percentage points $[+4.75, +5.32]$, 8/8 seeds, Holm-adjusted $p = 3.8\times10^{-9}$. Interval 5 reads 89.45 % ($+0.44$) and interval 20 reads 89.79 % ($+0.10$ $[-0.07, +0.27]$, 6/8, $p = 0.22$), so the cost again grows as the interval shortens and is no longer resolvable at 20. What changes is the scale: five points of top-1 on a 10.1 % error rate is half again the error, against 12 % of the zero baseline on UTKFace, so the same overwriting costs proportionally more when the metric is dense than a regression MAE implies.

**Attribution on that arm.** The straight-through control at interval 1 reads 90.03 %: $5.18$ points above the overwriting arm (0/8 the other way, Holm $p = 1.9\times10^{-11}$) and $0.14$ points above projection off (7/8, not resolved), which repeats the §5.4 attribution on a different task. The averaging order stays below resolution (interval 20, ema_after minus live: $-0.02$ points, 3/8, $p = 0.85$). Two limits apply: the images are the official split re-encoded as JPEG rather than the pickled arrays, so the pixels are not byte-identical to the reference release; and each arm takes its best epoch on the test split, which raises all six absolute accuracies by the same selection step without touching their differences — a re-run that holds 5,000 training images out for that choice and touches the test split once reads $+4.75$ points $[+4.31, +5.19]$ (8/8), so the selection step shifts the contrast by 0.28 points rather than creating it. The repeat covers the two endpoint arms only; intermediate intervals and 4-bit cells remain test-selected. Eight seeds leave a smallest detectable difference of 0.40 points at 8 bits, above the 0.2-point margin, so no equivalence is claimed here.

**Backends.** Measured as the share of the FP32 interval contrast that survives the compiled chain, ORT with static QDQ retains 1.039 and OpenVINO with NNCF 0.868 on the 200-image subset, against 0.950 for the vendor chain on the same 200 images. The three are not one experiment repeated three times — they differ in calibration set, operator coverage and rounding — so the ratio measures how much of the projection-off cost a pipeline can hide; the ordering contrast reads $+0.0109$ $[+0.0017, +0.0202]$, $+0.0007$ $[-0.0003, +0.0017]$ and $+0.0031$ $[+0.0028, +0.0034]$, and the negative control still reads $-0.0621$. The sign of the projection-off contrast is therefore not an artifact of one pipeline, but only the vendor chain's interval lies wholly below the 0.0016 bound, so only for it can an equivalence statement even be attempted (Text S4, Table S19).

**Subgroups.** The inferential unit remains the run; Table 5 reports deployment MAE by age band, sex and racial group over the same 1,556 images. The cost is not uniform — $+0.0328$ below age 20, 3.7 times the $+0.0088$ of the 20-39 band, and $+0.0306$ at 60 and above — while the 8 bit and FP32 columns agree within $\pm0.003$ in every group. The bands differ in size (304, 766, 297, 189 images) and an age band is not an independent group in the way a sex or race group is, so these are readings on one split and not population estimates.


**Table 5.** Deployment MAE by subgroup, pooled over eight seeds (1,556 images); cost is interval 1 minus projection off, in FP32.
| subgroup | images | FP32 | 8 bit | 8 bit $-$ FP32 | interval-1 cost |
|---|---|---|---|---|---|
| all | 1,556 | 0.0595 | 0.0595 | 0.0000 | $+0.0185$ |
| age below 20 | 304 | 0.0420 | 0.0416 | $-0.0004$ | $+0.0328$ |
| age 20–39 | 766 | 0.0478 | 0.0487 | $+0.0009$ | $+0.0088$ |
| age 40–59 | 297 | 0.0827 | 0.0827 | 0.0000 | $+0.0209$ |
| age 60 and above | 189 | 0.0985 | 0.0957 | $-0.0028$ | $+0.0306$ |
| male | 818 | 0.0620 | 0.0621 | $+0.0001$ | $+0.0163$ |
| female | 738 | 0.0566 | 0.0566 | 0.0000 | $+0.0209$ |
| White | 663 | 0.0643 | 0.0642 | $-0.0001$ | $+0.0207$ |
| Black | 315 | 0.0603 | 0.0604 | $+0.0001$ | $+0.0126$ |
| Asian | 223 | 0.0448 | 0.0449 | $+0.0001$ | $+0.0241$ |
| Indian | 253 | 0.0601 | 0.0602 | $+0.0002$ | $+0.0151$ |
| Other | 102 | 0.0562 | 0.0562 | 0.0000 | $+0.0180$ |

## 6. Discussion

Two mechanisms are at work and they are not of the same size. The interval between projections is the primary control: at seed 0 the protocol mean falls from 0.0743 at interval 1 to 0.0607 at interval 20, and switching projection off reads 0.0552; at the run level the two finite intervals sit 0.0185 and 0.0047 above projection off, in all eight paired runs and after Holm correction. Nothing here depends on grid alignment, since every finite-interval configuration ends up on the grid. The control that fake-quantizes without overwriting is indistinguishable from switching projection off, while at the same interval overwriting costs 0.0184 more, and the cost moves with the number of overwrites — which is why it is a curve rather than a single number.

The averaging order is the smaller control, and at this budget it is below what the experiment can resolve. The exported weights of the two orders differ by 0.043–0.091 in relative L2 against 0.64–0.83 for a change of seed, and requantizing both leaves 4.8 % ± 0.6 % with 98 % of the weights on different grid points; in function space the paired contrast is $+0.000029$, and the equivalence test rules out order effects of 0.0016 or larger at a margin fixed before the analysis. The ordering is therefore a sub-seed perturbation of the model's mean error, not of the error of a given image: across the eight paired runs the two orders rank the 1,556 images almost identically ($r = 0.97$–$0.99$) and their means differ by $+0.000029$, yet the absolute error of a single image moves by 0.005–0.010 on average (0.5–1.0 year), 13–37 % of the images move by more than a year, and the images that get worse are 47–52 % of the basis, so the two directions nearly balance (`quant_runs/order_perimage.csv`). What a 4.8 % weight difference buys is thus not a flat basin in function space but an error surface that is flat only after averaging; how the error is redistributed across images is not measured. The shadow is separate and smaller still: at interval 20 it lowers MAE by $-0.001377$ on the 1,556 held-out images, in all eight runs.

Both conclusions are conditional. A 60-epoch budget lowers the cost by a quarter and leaves it in all eight runs; a thirty-fold learning rate lowers it to 0.0055, which eight seeds cannot separate from zero, so the magnitude belongs to this budget rather than to the mechanism. The 6-bit readings track the FP32 ones and two independent implementations retain most of the interval effect, but the architecture and data axes and the cross-backend replication support its direction only, and the 4-bit ranking cannot be decided across seeds at all.

### 6.1 Deployment evaluation checklist

A team that reports a quantized deployment has to answer five questions in order. Table 6 turns them into a decision table, with the check that has to pass and the action it implies.

**Table 6.** A decision table for 8-bit weight-only deployment evaluation: the situation, the check that decides it, the action it implies and where the evidence is.

| Situation | Check before quoting a number | Action | Evidence |
|---|---|---|---|
| A fidelity score is about to be quoted | Operator, step and rounding of the two quantizers, then the residual against the limit of Proposition~\ref{prop:noinfo} ($5.0\times10^{-3}$ at four decimals, for $\kappa \approx 1$) | Below the limit, drop the score and report task MAE after export quantization; above it, report the residual or the cosine with provenance | §5.1, Proposition~\ref{prop:noinfo}, Algorithm~\ref{alg:precheck} |
| Models are about to be ranked by per-node cosine | Does the cosine ordering match the measured cost ordering across heads? | Do not rank by cosine; use the task metric after export quantization | §5.5 |
| Projection is to be kept or removed | Projection off against a longer interval; if projection must stay, STE against in-place overwrite | At 8 bits, switch projection off first: it costs 0.0185 MAE at interval 1 and 0.0047 at interval 20 against projection off. If projection must stay, prefer STE, and if overwriting is unavoidable replace nearest rounding by stochastic rounding | §5.2, §5.4 |
| No ordering effect is to be claimed | Margin, its provenance, the smallest detectable effect and the corrected p | State the smallest effect that is excluded, and report the margin and the correction next to the claim | §5.3 |
| A 4–6 bit ranking is to be reported | Cross-seed sd, paired t interval, median and bootstrap checks | Report all three; rank only where the paired difference against the reference excludes zero under the pre-specified rule, never by whether two marginal intervals overlap | §5.5 |

A worked case fixes the order. Suppose a vendor report quotes a cosine of 0.999 for an int8 export of this network. The first check decides before the number is read: with the deployment quantizer recomputing the scale from the exported tensor, the residual stays inside the limit of Proposition 1, so recomputing the cosine gives 1.0000. A quoted 0.999 therefore cannot come from that comparison at all: it measures a different pair of tensors, and the check names the pair before the number is trusted. The second check replaces it with an ordering question, and what gets reported is the task MAE after export quantization. The third then asks whether projection can be switched off, which is a change to the training loop rather than to the report. Algorithm 1 is the executable form of the first check.

The rules are conditional in one more way: they describe what a number means, not which model to ship. The choice of architecture, dataset and training budget is outside them, and nothing here licenses deploying an age regressor in an access-control or identity-checking role (§9).

## 7. Conclusion

A team holding an int8 model has to answer two questions: which weights to export, and which fidelity numbers to trust. On the first, the interval between projections matters and the averaging order does not. Projection off is best, projecting every step costs 0.0185 MAE, every twentieth step 0.0047, and the two averaging orders differ by 0.000029 against a pre-specified bound of 0.0016. On the second, at 8 bits the export quantizer is a no-op: the centered cosine reads 1.0000 for every protocol and cannot rank them, and the resolution bound of Proposition 1 says when that reading is inevitable rather than accidental.

The result that transfers is the protocol rather than a new quantizer or a budget-independent law. The five rules of Table 6 order the checks a report should pass before it quotes a fidelity score, Proposition 1 and Algorithm 1 make the first of them runnable, and the same checks hold on a vendor NPU chain and on two open-source runtimes. The magnitudes belong to this budget — tripling the budget or raising the learning rate moves them — and the architecture, data and task axes corroborate the direction only, while 4 bits cannot be ranked across seeds at all. Two follow-ups the protocol asked for are now in hand: the mechanism model of §5.4 prices a single overwrite, and the classification arm of §5.5 reads the interval-1 cost as five points of top-1, which the straight-through control removes.

## 8. Threats to validity

The two zero-cost controls locate the cost on the overwrite rather than on the grid (§5.4), but neither records the updates that disappear: per-weight optimizer moments, the share of updates below half a grid step and the update norm were not measured, so the attribution is consistent with the overwrite rather than a direct demonstration of it.

Threats are numbered 1–14 once for the whole paper: items 1–8 are stated here, except 3, 5 and 6, which are carried in Text S6.1 under the same numbers; items 9–14, which change only how an individual number is read, are in Text S6.2.

1. One dataset: UTKFace, small, curated and near-frontal, with the 4.41 M-parameter network only. §5.5 adds endpoint comparisons on ResNet-18 and KonIQ-10k, 8 initial seeds each, which keep the direction; at 1e-4 the architecture axis covers four intervals and all three protocols, the data axis two endpoints only, so other domains, resolutions and architectures are untested beyond them. Every run-level contrast is also measured at one fixed split, so its interval carries initialization noise and not split noise.

2. Throughout §5.1–§5.5 the controlled variable is per-output-channel symmetric weight QDQ in PyTorch, independent of a compiler's heuristics. §5.5 adds the vendor toolchain on five checkpoints across three heads through a 100-image calibration set, and ONNX Runtime on the 66 convolutions and OpenVINO at mixed precision on the relu_qat head, two endpoints and 200 test images.

4. Checkpoint selection overlaps the evaluation subset: the best epoch is chosen on the 2,000-image monitoring subset of each split, which the 400-image deployment subset overlaps, so those readings are optimistic. The main basis is the 1,556 images that took no part in selection: the shadow contrast at interval 20 is 8/8 with $p = 0.0078$ there against 7/8 and $p = 0.070$ on the 400, and contrast a reads $+0.018457$ against $+0.017750$, contrast b $+0.004723$ against $+0.003800$.

7. Every reading of §5.2–§5.5 uses the per-channel max grid that training also uses. Moving the same checkpoints onto per-tensor max and per-channel p99 clipping keeps the direction ($+0.0192$ / $+0.0181$ / $+0.0213$), but p99 alone raises the absolute error by 0.0441–0.0544, 2.1–2.6 times the interval-1 effect and an order above the interval-20 effect ($+0.0038$): only the direction transfers.

8. The four control-head checkpoints of Table S15 are seed 0 only, so the spread across heads ($+0.0038$ / $+0.0034$ / $-0.0002$ / $+0.0042$) is descriptive and cannot be resolved into head effects; the run-level vendor reading that does carry eight seeds covers the relu_qat head at two endpoints only.

## 9. Data, Ethics and Broader Impact

**Data and ethics.** The aligned, cropped UTKFace subset [36], taken on 2026-10-26 from a HuggingFace mirror, holds 23,708 JPEG images (SHA-256 over name-ordered filenames with byte sizes: 493950cf…). The official page says "non-commercial research purposes only" and the mirror MIT, so we follow the stricter, and we read only the age label released with the images. The seeded 15 % split carries no identity labels, so one person may fall on both sides and no number here is evidence about unseen identities.

The part of that leakage which can be measured is duplication of photographs. The corpus holds 390 byte-identical copies in 375 groups (at most four copies of one image), and under the split used here 44 of the 1,556 external images (2.8 %) are byte-identical to a training image while 56 (3.6 %) have a near-duplicate under a 64-bit gradient hash followed by a pixel check (median Hamming distance 6 bits against 26 for unrelated pairs). Recomputing the four contrasts on the same eight seeds without those 56 images moves them by at most 0.00013 (2.7 % of contrast b): a $+0.018457$ to $+0.018538$, b $+0.004723$ to $+0.004852$, c $-0.001377$ to $-0.001466$, d $+0.000029$ to $+0.000031$, with a, b and c keeping 8/8 of one sign and d inside the pre-specified margin. Duplicates therefore do not drive the readings; what remains unmeasured is one person photographed differently, which the released labels cannot separate.

**Broader impact.** Age regression feeds targeting, access control and identity checks, and an error lands on the person judged. Table 5 reports the error by age band, sex and race for this dataset: the cost of projection is 0.0328 below 20 and 0.0088 in 20–39, so a figure pooled over all images understates the loss for the youngest subjects. These are readings on 1,556 images under one split, not population estimates. The contribution is evaluation only, with no detector, rejection rule or correction; nothing here prevents those uses, and nothing here supports deploying them.

## 10. Reproducibility and compute

Every number here is produced by a script in this repository, and Table S13 in Text S6 (S6.3) maps each claim to its artifact, script and figure or table. The scripts and result files quoted here are those of the released snapshot, and `manifest.json` lists each of them with its SHA-256. Training one run takes 834–891 s on a single RTX 4080 Laptop (7,875 MiB free) at 20 epochs and batch 64, and the four additional runs of the 3 × 3 design take about 57 min in total. The 40 main-sequence runs are therefore 9.3–9.9 device hours on that one machine; the other families are no longer counted only as runs: summing the per-epoch `sec` column of each run's `history.csv` gives 84.8 device hours over the 359 runs of the released snapshot, of which the main-sequence family — those 40 runs plus the later top-ups, 79 runs — is 21.2 h, the ResNet-18 architecture axis 15.4 h and the CIFAR-10 arm 12.2 h (`quant_runs/compute_table.csv`; Text S2 gives the remaining eight families). That column covers the training step and the monitoring evaluation of the same epoch, while export quantisation, compilation and the held-out evaluation fall outside it.

```latex
\section*{Acknowledgements}
The authors thank the maintainers of the UTKFace and KonIQ-10k benchmarks for releasing them and for documenting their licence terms. No external funding was received for this work.
```


## Declarations

### Funding

This work received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

### Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

### Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the authors used large language model assistants to draft and edit the manuscript and to check the analysis code against the reported numbers. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

### Ethics approval and consent to participate

Not applicable. The study is a secondary analysis of two publicly available benchmark datasets (UTKFace and KonIQ-10k); no participants were recruited and no new human data were collected.

### Consent for publication

Not applicable.

### Data availability

The UTKFace and KonIQ-10k datasets are publicly available from their original distributors and are not redistributed here because of their license terms; the exact files, splits and checkpoints used are recorded in the anonymized repository https://github.com/una393/utkface-snap-order, together with the scripts that regenerate every reported number. Table S13 of the Supplementary Material maps each claim to its artifact.

### Code availability

All training, quantization, compilation and analysis code, and the derived result files, are available in the same anonymized repository (703 files, about 10 MB, with `manifest.json` giving per-file SHA-256). The code is released under the MIT license; data and weights follow the upstream non-commercial terms, as stated in the NOTICE file. Algorithm 1 is also packaged: `pip install .` installs a `fidelity-check` entry point that takes a checkpoint and a deployment quantizer and prints the branch the model falls in, and running it from the installed package reproduces the six verdicts of Table S7.

### CRediT authorship contribution statement

**Anonymous Author(s):** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review and editing, Visualization. All authors read and approved the final manuscript.


## 10. References

[1] Bengio Y, Léonard N, Courville A (2013) Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432
[2] Choi J, Wang Z, Venkataramani S, Chuang PI-J, Srinivasan V, Gopalakrishnan K (2018) PACT: parameterized clipping activation for quantized neural networks. arXiv preprint arXiv:1805.06085
[3] Esser SK, McKinstry JL, Bablani D, Appuswamy R, Modha DS (2020) Learned step size quantization. In: International conference on learning representations (ICLR). arXiv:1902.08153
[4] Zhou A, Yao A, Guo Y, Xu L, Chen Y (2017) Incremental network quantization: towards lossless CNNs with low-precision weights. In: International conference on learning representations (ICLR). arXiv:1702.03044
[5] Zhao K, Tabaru T, Kobayashi K, Honda T, Yamazaki M, Tsuruoka Y (2025) Direct quantized training of language models with stochastic rounding. In: Proceedings of the 17th Asian conference on machine learning (ACML), PMLR vol 304, pp 1150–1165
[6] Nikdan M, Zandieh A, Alistarh D, Mirrokni V (2026) ECO: quantized training without full-precision master weights. arXiv preprint arXiv:2601.22101
[7] Jacob B, Kligys S, Chen B, Zhu M, Tang M, Howard A, Adam H, Kalenichenko D (2018) Quantization and training of neural networks for efficient integer-arithmetic-only inference. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp 2704–2713. https://doi.org/10.1109/CVPR.2018.00286
[8] Krishnamoorthi R (2018) Quantizing deep convolutional networks for efficient inference: a whitepaper. arXiv preprint arXiv:1806.08342
[9] Nagel M, Fournarakis M, Amjad RA, Bondarenko Y, van Baalen M, Blankevoort T (2021) A white paper on neural network quantization. arXiv preprint arXiv:2106.08295
[10] Lee J, Jeon J, Kim D, Ham B (2025) Scheduling weight transitions for quantization-aware training. In: Proceedings of the IEEE/CVF international conference on computer vision (ICCV). arXiv:2404.19248
[11] Polyak BT, Juditsky AB (1992) Acceleration of stochastic approximation by averaging. SIAM J Control Optim 30:838–855. https://doi.org/10.1137/0330046
[12] Tarvainen A, Valpola H (2017) Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results. In: Advances in neural information processing systems (NIPS), vol 30. arXiv:1703.01780
[13] Yang G, Zhang T, Kirichenko P, Bai J, Wilson AG, De Sa C (2019) SWALP: stochastic weight averaging in low-precision training. In: Proceedings of the international conference on machine learning (ICML). arXiv:1904.11943
[14] Shin S, Boo Y, Sung W (2021) SQWA: stochastic quantized weight averaging for improving the generalization capability of low-precision deep neural networks. In: Proceedings of the IEEE international conference on acoustics, speech and signal processing (ICASSP). arXiv:2002.00343
[15] Izmailov P, Podoprikhin D, Garipov T, Vetrov D, Wilson AG (2018) Averaging weights leads to wider optima and better generalization. In: Proceedings of the conference on uncertainty in artificial intelligence (UAI). arXiv:1803.05407
[16] Wortsman M, Ilharco G, Gadre SY, Roelofs R, Gontijo-Lopes R, Morcos AS, Namkoong H, Farhadi A, Carmon Y, Kornblith S, Schmidt L (2022) Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In: Proceedings of the international conference on machine learning (ICML). arXiv:2203.05482
[17] Li Y, Shen M, Ma J, Ren Y, Zhao M, Zhang Q, Gong R, Yu F, Yan J (2021) MQBench: towards reproducible and deployable model quantization benchmark. In: Proceedings of the neural information processing systems track on datasets and benchmarks (NeurIPS D&B). arXiv:2111.03759
[18] Xiao G, Lin J, Seznec M, Wu H, Demouth J, Han S (2023) SmoothQuant: accurate and efficient post-training quantization for large language models. In: Proceedings of the international conference on machine learning (ICML). arXiv:2211.10438
[19] Nagel M, Amjad RA, van Baalen M, Louizos C, Blankevoort T (2020) Up or down? Adaptive rounding for post-training quantization. In: Proceedings of the international conference on machine learning (ICML). arXiv:2004.10568
[20] Li Y, Gong R, Tan X, Yang Y, Hu P, Zhang Q, Yu F, Wang W, Gu S (2021) BRECQ: pushing the limit of post-training quantization by block reconstruction. arXiv preprint arXiv:2102.05426
[21] Dong Z, Yao Z, Gholami A, Mahoney M, Keutzer K (2019) HAWQ: Hessian AWare quantization of neural networks with mixed-precision. In: Proceedings of the IEEE/CVF international conference on computer vision (ICCV), pp 293–302. https://doi.org/10.1109/ICCV.2019.00038
[22] Dong Z, Yao Z, Cai Y, Arfeen D, Gholami A, Mahoney MW, Keutzer K (2019) HAWQ-V2: Hessian aware trace-weighted quantization of neural networks. arXiv preprint arXiv:1911.03852
[23] Frantar E, Ashkboos S, Hoefler T, Alistarh D (2023) GPTQ: accurate post-training quantization for generative pre-trained transformers. In: International conference on learning representations (ICLR). arXiv:2210.17323
[24] Lin J, Tang J, Tang H, Yang S, Chen W-M, Wang W-C, Xiao G, Dang X, Gan C, Han S (2024) AWQ: activation-aware weight quantization for LLM compression and acceleration. In: Proceedings of the conference on machine learning and systems (MLSys). arXiv:2306.00978
[25] Ashkboos S, Mohtashami A, Croci ML, Li B, Cameron P, Jaggi M, Alistarh D, Hoefler T, Hensman J (2024) QuaRot: outlier-free 4-bit inference in rotated LLMs. In: Advances in neural information processing systems (NeurIPS), vol 37. arXiv:2404.00456
[26] Liu Z, Zhao C, Fedorov I, Soran B, Choudhary D, Krishnamoorthi R, Chandra V, Tian Y, Blankevoort T (2025) SpinQuant: LLM quantization with learned rotations. In: International conference on learning representations (ICLR). arXiv:2405.16406
[27] Zhou S, Wu Y, Ni Z, Zhou X, Wen H, Zou Y (2016) DoReFa-Net: training low bitwidth convolutional neural networks with low bitwidth gradients. arXiv preprint arXiv:1606.06160
[28] Courbariaux M, Bengio Y, David J-P (2015) BinaryConnect: training deep neural networks with binary weights during propagations. arXiv preprint arXiv:1511.00363
[29] Wang N, Choi J, Brand D, Chen C-Y, Gopalakrishnan K (2018) Training deep neural networks with 8-bit floating point numbers. arXiv preprint arXiv:1812.08011
[30] Bai Y, Wang Y-X, Liberty E (2019) ProxQuant: quantized neural networks via proximal operators. In: International conference on learning representations (ICLR). arXiv:1810.00861
[31] Louizos C, Reisser M, Blankevoort T, Gavves E, Welling M (2018) Relaxed quantization for discretized neural networks. arXiv preprint arXiv:1810.01875
[32] Gholami A, Kim S, Dong Z, Yao Z, Mahoney MW, Keutzer K (2022) A survey of quantization methods for efficient neural network inference. In: Low-power computer vision. Chapman and Hall/CRC, pp 291–326. https://doi.org/10.1201/9781003162810-13. arXiv:2103.13630
[33] Yin P, Zhang S, Lyu J, Osher S, Qi Y, Xin J (2018) BinaryRelax: a relaxation approach for training deep neural networks with quantized weights. SIAM J Imaging Sci 11:2205–2223. https://doi.org/10.1137/18m1166134
[34] Radosavovic I, Kosaraju RP, Girshick R, He K, Dollár P (2020) Designing network design spaces. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp 10425–10433. https://doi.org/10.1109/CVPR42600.2020.01044
[35] Wightman R (2019) PyTorch image models. GitHub repository. https://github.com/rwightman/pytorch-image-models. https://doi.org/10.5281/zenodo.4414861
[36] Zhang Z, Song Y, Qi H (2017) Age progression/regression by conditional adversarial autoencoder. In: Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pp 4352–4360. https://doi.org/10.1109/CVPR.2017.463
[37] Wu Y, He K (2018) Group normalization. In: Proceedings of the European conference on computer vision (ECCV), LNCS vol 11217, pp 3–19. https://doi.org/10.1007/978-3-030-01261-8_1
[38] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Ł, Polosukhin I (2017) Attention is all you need. In: Advances in neural information processing systems (NIPS), vol 30. arXiv:1706.03762
[39] Zhang B, Sennrich R (2019) Root mean square layer normalization. In: Advances in neural information processing systems (NeurIPS), vol 32. arXiv:1910.07467
[40] Schuirmann DJ (1987) A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. J Pharmacokinet Biopharm 15:657–680. https://doi.org/10.1007/BF01068419
[41] Lakens D, Scheel AM, Isager PM (2018) Equivalence testing for psychological research: a tutorial. Adv Methods Pract Psychol Sci 1:259–269. https://doi.org/10.1177/2515245918770963
[42] Gupta S, Agrawal A, Gopalakrishnan K, Narayanan P (2015) Deep learning with limited numerical precision. In: Proceedings of the international conference on machine learning (ICML), pp 1737–1746. arXiv:1502.02551
[43] Ozkara K, Yu T, Park Y (2025) Stochastic rounding for LLM training: theory and practice. In: Proceedings of the international conference on artificial intelligence and statistics (AISTATS), PMLR vol 258, pp 4402–4410
[44] Hosu V, Lin H, Sziranyi T, Saupe D (2020) KonIQ-10k: an ecologically valid database for deep learning of blind image quality assessment. IEEE Trans Image Process 29:4041–4056. https://doi.org/10.1109/TIP.2020.2967829
