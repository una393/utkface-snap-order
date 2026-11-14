# Frozen analysis plan for the confirmatory contrasts

**Status.** The analysis plan behind the six confirmatory contrasts of the paper was fixed
before the run-level readings on the external subset were computed. It was **pre-planned,
not externally registered**: there is no OSF or AsPredicted entry, no registry identifier,
and no registry timestamp. This file transcribes the plan and dates it by the commit that
first contains it; that commit date is the only timestamp it has, and the paper says so.

Everything below is a record of what was decided. Where practice later departed from the
plan, the departure is listed in §5 with the numbers that changed, so that a reader can
recompute both the planned and the delivered reading.

## 1. What the plan fixed

| Item | Frozen value |
|---|---|
| Task | UTKFace age regression, target = age / 100, 23,708 images |
| Recipe | symmetric per-output-channel int8, one quantizer, scale `s = max_j abs(W_oj)/127`, round-to-nearest, no clipping |
| Trainable copy | in place: no full-precision master copy at the optimizer (the EMA shadow is an export copy) |
| Grid projection | every `N` steps; `N = 1` every step, `N = infinity` never |
| Shadow | `m <- gamma*m + (1-gamma)*W`, `gamma = 0.99`, updated once per optimizer step |
| Protocols | `live` (no shadow, export the last projected iterate), `ema_before` (average unprojected iterates, project once at export), `ema_after` (average projected iterates) |
| Training budget | 20 epochs, batch 64, Adam, AMP, gradient clip 1.0 |
| Split | `split_random(val_ratio = 0.15, seed = 0)`, fixed for every confirmatory run; only the initialization varies |
| Primary outcome | deployment MAE: the exported weights evaluated as they are (FP32) and after re-quantization to 8 bits |
| Inference unit | the run. Eight paired seeds per cell, `df = 7`; images are never treated as replicates |

## 2. The six confirmatory contrasts

Positions are run-level; the label is the contrast, not the interval.

| # | Contrast | Expected sign | Basis |
|---|---|---|---|
| a | `live` at interval 1 minus `live` with projection off | positive (projection costs) | FP32 deployment MAE |
| b | `live` at interval 20 minus `live` with projection off | positive | FP32 deployment MAE |
| c | `ema_after` minus `live`, both at interval 20 | negative (shadow helps) | FP32 deployment MAE |
| d | `ema_before` minus `ema_after`, both at interval 20 | none stated; equivalence is the claim | FP32 deployment MAE |
| e | STE (forward fake-quantization, no overwrite) minus in-place overwriting at interval 1 | negative (the control is cheaper) | 8 bit deployment MAE |
| f | STE minus projection off | none stated; equivalence is the claim | 8 bit deployment MAE |

Contrasts a-d were published on the FP32 column of `quant_runs/heldout_eval_v3.csv`; the
8 bit column is reported alongside and never substitutes for it. Contrasts e and f are read
at 8 bits because the STE arm exists only there.

## 3. Decision rules fixed with the plan

- **Test.** Two-sided paired t over the eight seed pairs, `df = 7`, `alpha = 0.05`, Holm
  correction inside the family of six. No other correction is applied to these six.
- **Equivalence.** TOST with a margin of 0.0016, fixed before the readings: 1 % of the zero
  baseline of the same split (0.15526728380), i.e. 0.16 years, two orders of magnitude below
  the interval-1 cost. It applies to contrast d only, and only on the 8 bit basis. The
  smallest detectable effect at 80 % power is 0.000383, so a bound of 0.0016 is meaningful
  for d and is not available at 6 bits (their MDE is 0.0011-0.0027).
- **Wording.** A contrast that survives neither a difference test nor an equivalence test is
  reported as "not detected at this resolution", never as zero.
- **Scope of the claims.** Confirmatory conclusions are limited to one dataset, one recipe,
  one 20-epoch budget and 8 bits.

## 4. Exploratory by construction

The 6 bit and 4 bit bases, the 5-epoch-interval middle cells, the gamma sweeps, the second
architecture (ResNet-18), the second dataset (KonIQ-10k), the third deployment backend
(vendor hbdk4), the cross-backend ratios and the subgroup breakdown. None of these carries a
confirmatory claim, however small its p value.

## 5. Departures made after the plan

Every row is recomputed by `scripts/topup_table.py` into `quant_runs/topup_table.csv`.

| Contrast | First reading (seeds 0-2) | Delivered (seeds 0-7) | Change |
|---|---|---|---|
| a, interval 1 vs off | +0.018687 (3/3) | +0.018457 (8/8) | -0.000231 |
| b, interval 20 vs off | +0.004619 (3/3) | +0.004723 (8/8) | +0.000104 |
| c, shadow at interval 20 | -0.001730 (0/3) | -0.001377 (0/8) | +0.000352 |
| d, order at interval 20 | +0.000081 (2/3) | +0.000029 (5/8) | -0.000052 |
| e, STE vs overwrite | -0.018307 (0/3) | -0.018441 (0/8) | -0.000134 |
| f, STE vs off | +0.000328 (2/3) | -0.000097 (4/8) | -0.000424 |
| a, 60-epoch budget | +0.014006 (3/3) | +0.013801 (8/8) | -0.000205 |
| a, 30x learning rate | +0.006591 (3/3) | +0.005466 (6/8) | -0.001125 |

1. **Seeds.** Seeds 3-7 were added to the five main sequences and to the STE arm after the
   first reading, taking every cell from three seeds to eight. No sign flips. Contrast c
   shrinks by 25 % of its own size and stays 8/8 negative. Contrast f moves across zero
   (+0.000328 to -0.000097) but was never a difference claim: both readings are inside the
   0.0016 bound. The learning-rate row weakens from 3/3 to 6/8; it is exploratory, and the
   paper reports the weaker sign count.
2. **Basis.** The confirmatory contrasts moved from the 400-image monitor subset to the
   1,556-image external subset, because the 400 images took part in checkpoint selection.
   On 400 images the same four contrasts read a +0.017773, b +0.003808, c -0.001165 (7/8
   negative) and d +0.000183; the basis change moves each estimate by at most 0.0009 and
   changes no sign.
3. **6 bit.** Four contrasts at 6 bits were corrected post hoc as a family of their own
   (Holm, four tests). The correction is reported but does not make them pre-specified, and
   no equivalence is claimed there.
4. **Architecture axis.** At 1e-4 the sweep was extended by two cells (interval 5 and
   `ema_before` at interval 20), from 48 to 72 runs. The axis is a paired replication, not
   a confirmatory test; at 1e-4 the network is far from converged (0.1290 at interval 1
   against 0.0564 with projection off), so its absolute levels are read as directional.
5. **Third backend.** The vendor toolchain and the two open-source runtimes were run after
   the plan and enter only as deployment evidence, not as hypothesis tests.
6. **Duplicate audit.** Forty-four byte-identical and 56 near-duplicate images were removed
   from the held-out subset as a robustness check. The four contrasts then read
   a +0.018538, b +0.004852, c -0.001466 (8/8) and d +0.000031: the largest change is
   0.00013, 2.7 % of b, and no sign moves.

## 6. What this file is not

It is not a registry entry and must not be cited as one. It is the plan as it stood, plus
every departure, in a form a reader can check against the artifacts below. The paper cites
it as "the frozen plan (`quant_runs/preregistration_freeze.md`, commit <hash>)".

## 7. Artifacts

- `quant_runs/heldout_eval_v3.csv` - per-run deployment MAE on the 400- and 1,556-image subsets.
- `quant_runs/topup_table.csv` - the eight rows of §5, regenerated by `scripts/topup_table.py`.
- `quant_runs/budget_contrast.json` - per-seed differences behind the budget and learning-rate rows.
- `quant_runs/paired_report.md` - the paired differences behind a-d and their intervals.
- `quant_runs/fidelity_check.json` - the admissibility pre-check of Algorithm 1.
