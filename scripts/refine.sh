#!/usr/bin/env bash
# Optional confidence-weighted edge-matching refinement of the floor0 pose.
# Default: a SINGLE config — 6-DOF ("full") refinement, trust-region anchor 300, edge weighting =
# edge-probability x confidence. This is the safe, near-best global setting: a small, regime-dependent
# gain over selection-only, and it does not hurt (anchor 300 ~ identity). No sweep, no per-scene tuning.
# Set SWEEP=1 to instead run the full {full,transl} x anchor-{10,30,300} exploration (writes 6 CSVs).
#   M3ED:  SEQ=falcon_indoor_flight_3 bash scripts/refine.sh
#   DSEC:  DATASET=dsec SEQ=thun_00_a bash scripts/refine.sh
# Uses the eval dump (scripts/eval.sh) as the init poses. ITERS=24 (default) or 12.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/scenes.sh
DATASET="${DATASET:-m3ed}"; ITERS="${ITERS:-24}"
: "${SEQ:?set SEQ (test sequence)}"
OD="outputs/refine"; mkdir -p "$OD"
INIT="${INIT:-outputs/eval/${SEQ}__iter${ITERS}.npz}"
[ -f "$INIT" ] || { echo "missing eval dump $INIT -- run scripts/eval.sh first"; exit 1; }

if [ "$DATASET" = "dsec" ]; then
  CKPT="${CKPT:-$(find "$CKPT_ROOT/dsec" -name 'best_model.pth' 2>/dev/null | head -1)}"
  RT="test_set_init_pose/DSEC/test_RT_seq_${SEQ}_5.00_0.50.csv"; DEPTH=""; DS=dsec
else
  CKPT="${CKPT:-$(find "$CKPT_ROOT/$SEQ" -path '*partial_light*' -name 'best_model.pth' 2>/dev/null | head -1)}"
  RT="test_set_init_pose/M3ED/test_RT_seq_${SEQ}_5.00_0.50.csv"; DEPTH="--partial_depth partial_light"; DS=m3ed
fi
: "${CKPT:?no checkpoint found (set CKPT=...)}"
PREF="$OD/${SEQ}_iter${ITERS}_floor0"
BASE=(--seq "$SEQ" --dataset "$DS" --load_checkpoints "$CKPT" $DEPTH --test_rt "$RT" \
      --iters "$ITERS" --init_poses "$INIT" --select floor0)

if [ "${SWEEP:-0}" = "1" ]; then
  echo "=== REFINE SWEEP $DS $SEQ | iters=$ITERS (6 configs) ==="
  $PY tools/pose_refine/eval_refine.py "${BASE[@]}" --sweep_both --sweep_src both --out_prefix "$PREF"
  echo "=== sweep done -> ${PREF}__both__{full,transl}__aw{10,30,300}.csv ==="
else
  echo "=== REFINE $DS $SEQ | iters=$ITERS | full, anchor 300, edge-prob x confidence ==="
  $PY tools/pose_refine/eval_refine.py "${BASE[@]}" \
    --edge_conf weight --edge_conf_src both --anchor_w 300 --out "${PREF}_refined.csv"
  echo "=== refined -> ${PREF}_refined.csv ==="
fi
