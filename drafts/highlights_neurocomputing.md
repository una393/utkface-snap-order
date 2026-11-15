# Highlights

- In-place int8 projection costs +0.0185 MAE per step and +0.0047 at interval 20.
- A no-overwrite STE control shows no cost; stochastic rounding removes 99.8% of it.
- At 8 bits, export re-quantization changes nothing; centered cosine stays 1.0.
- Proposition 1 and Algorithm 1 gate fidelity scores before they are quoted.
- Five-rule decision table tested on a target NPU board and two open-source runtimes.
