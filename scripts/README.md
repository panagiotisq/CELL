# CELL automation scripts

These reproduce the paper's per-sequence pipeline with the **exact** training/eval/refinement
recipe. Run them from the CELL repo root. Geometry (crop / calibration / `max_depth`) is resolved
automatically per sequence by `core/scene_config.py`; you pass only the sequence.

| script | what it does |
| :-- | :-- |
| `scenes.sh` | scene registry (test↔train splits, event representation). Sourced by the others; edit `PY`/`M3ED_DATA`/`DSEC_DATA`/`CKPT_ROOT` via env. |
| `train.sh` | train one model — M3ED per scene (partial depth completion + decoupled β1.5 confidence), or the single DSEC model (sparse depth). |
| `eval.sh` | eval on the shipped test perturbation; reports `all`/`floor0`/`floor0.5`/`>median`; `--save_poses` dump for refinement. `ITERS=24` or `12`. |
| `refine.sh` | optional edge-matching refinement of the `floor0` pose — one config by default (full, anchor 300, edge-prob×conf); `SWEEP=1` runs the full grid. |
| `run_all.sh` | orchestrates train→eval→refine over all scenes and both iteration budgets. |

## Examples
```bash
# one M3ED scene, full pipeline
SEQ=falcon_indoor_flight_3 bash scripts/train.sh
SEQ=falcon_indoor_flight_3 bash scripts/eval.sh         # 24-iter; ITERS=12 for the halved budget
SEQ=falcon_indoor_flight_3 bash scripts/refine.sh

# DSEC: one model, then every test sequence
DATASET=dsec bash scripts/train.sh
DATASET=dsec SEQ=thun_00_a bash scripts/eval.sh
DATASET=dsec SEQ=thun_00_a bash scripts/refine.sh

# everything (long); or just re-eval with existing checkpoints
bash scripts/run_all.sh
STAGES="eval refine" bash scripts/run_all.sh
```

## Deployed choices (as in the paper)
- **Selection = `floor0`** (probabilistic keep w.p. = normalized confidence). `eval.sh` reports all four
  rules; `floor0` is the headline/refinement init.
- **M3ED** uses `partial_light` completed depth as the encoder input **and** as the flow-supervision
  support — during training the flow loss is computed on that completed support (`--dense_flow`), and the
  same `partial_light` depth is used at eval. **DSEC** uses sparse depth (one model).
- **Refinement** is a small, optional add-on over `floor0`. The default is a single global config (6-DOF,
  trust-region anchor 300, edge-probability×confidence weighting) that does not hurt; `SWEEP=1` runs the full
  {full,transl}×anchor grid for analysis.

## SLURM
To run on a cluster, wrap any of these in a thin submit script, e.g.:
```bash
#!/bin/bash
#SBATCH --gres=gpu:1 -t 48:00:00
SEQ=$SEQ bash scripts/train.sh
```
and `sbatch --export=ALL,SEQ=falcon_indoor_flight_3 train.slurm`.
