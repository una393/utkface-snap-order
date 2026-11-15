# Weight-only faithfulness metrics have saturated at 8 bits

Code, manuscript sources and reported tables for a controlled measurement of
**weight-grid projection, shadow (EMA) writing and averaging order** in
quantization-aware training (QAT), on UTKFace age regression.

The paper's claim in one line: once training-time weights are projected onto a
fixed symmetric per-output-channel int8 grid, how *often* you project and how
you *order* projection versus EMA do not act on the usual axis — switching the
projection off is worth about $+0.018$ MAE, but the cheapest documented fix is
to replace the single nearest-rounding step with **stochastic rounding**, which
recovers 99.8 % of that cost at zero extra cost.

## Headline numbers

Held-out split, 1,556 images, 8-bit weight-only, 8 initial seeds per sequence
(fixed data split), paired per seed, df = 7:

| Contrast | Effect | 95 % CI | sign |
|---|---|---|---|
| interval 1 vs projection off | +0.018457 | [+0.017498, +0.019415] | 0/8 negative |
| interval 20 vs projection off | +0.004723 | [+0.004028, +0.005417] | 0/8 |
| shadow on - off @ interval 20 | -0.001377 | [-0.001798, -0.000957] | 8/8 |
| before- vs after-snap @ 20 | +0.000029 | [-0.000249, +0.000307] | 3/8; TOST p < 1e-4 at a pre-registered margin of 0.0016 |
| stochastic vs nearest rounding (interval 1) | -0.018306 | [-0.019369, -0.017243] | 8/8 |

Cross-backend replication (200 images, one checkpoint pair each, relative
retention of the interval effect after int8): vendor NPU toolchain 0.950,
ONNX Runtime QDQ 1.039, OpenVINO + NNCF 0.868.

## Layout

| Path | Contents |
|---|---|
| `train.py` | single training run (`--ema-mode {none,before_snap,after_snap}`, `--snap-every`, `--quant-mode {overwrite,ste}`) |
| `models/`, `quant/` | RegNetY-008 backbone, control heads, grid projection and EMA shadow code |
| `scripts/` | experiment drivers, statistics, figure builders, docx/PDF builder |
| `quant_runs/` | reported tables (`.csv`), statistics (`.json`), per-image predictions (`.npz`) |
| `npu_quant/` | vendor-toolchain (`hbdk4`) compile configs, calibration lists, node-wise cosine reports |
| `drafts/` | manuscript sources: `paper2_snap_order_zh.md` (Chinese, source of truth) and `paper2_snap_order.md` (English mirror), plus built `.docx`/`.pdf` and `figs/` |

## Data

Not included (see `NOTICE`). To rebuild:

```bash
# UTKFace from the HuggingFace mirror py97/UTKFace-Cropped -> 23,708 jpgs
python scripts/cache_raw.py --src D:/data/utkface/UTKFace --out data/utkface_raw224.npy
```

Split: `split_random(val_ratio=0.15, seed=0)`; the held-out 1,556 images are
disjoint from both the 2,000-image checkpoint-selection monitor set and the
400-image deployment subset.

## Reproduce

Verified with `torch 2.12.0+cu132`, `onnxruntime 1.26.0`, `numpy 2.4.6`,
`openvino 2026.4.1`, `nncf 3.4.0` (`requirements.txt`).

```bash
# one training run (~12-15 min on an RTX 4080 Laptop)
python train.py --raw data/utkface_raw224.npy --head relu_qat --ema-mode after_snap \
                --snap-every 20 --epochs 20 --batch-size 64 --amp --grad-clip 1.0 \
                --ema-decay 0.99 --seed 0 --split-seed 0

# deployment table from the checkpoints, then the run-level statistics
python scripts/deploy_table.py --out quant_runs/deploy_table_v3.csv --dump quant_runs/perimage
python scripts/round20_table_v4.py     # common 400-image basis -> deploy_table_v4.csv
python scripts/round20_heldout.py      # held-out 1,556 basis   -> heldout_eval_v3.csv
python scripts/round20_runlevel.py     # seed-paired contrasts   -> runlevel2_stats.json
python scripts/round20_dual_basis.py   # both bases side by side -> dual_basis_report.json
python scripts/round20_mechanism.py    # learning rate / budget / stochastic rounding
python scripts/round20_backend2.py     # ONNX Runtime QDQ
python scripts/round20_backend_ov.py   # OpenVINO + NNCF
```

Each script writes its JSON/Markdown next to the table it reads, so the reported
numbers can be re-derived without the checkpoints being present.

## Manuscript build (Chinese, two-column journal layout)

```bash
python scripts/_publish_journal.py     # md -> docx -> PDF (Word COM), asserts page/word count
python scripts/_verify_xref.py         # figure/table cross-references, equation numbering, layout
```
Requires `python-docx`, `latex2mathml`, `pymupdf` and a local Word installation.

## License

Code and manuscript text: MIT (`LICENSE`). Data, pretrained weights and
compiled artifacts are not distributed and keep their upstream terms
(`NOTICE`).
