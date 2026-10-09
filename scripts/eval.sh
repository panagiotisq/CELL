#!/usr/bin/env bash
# Evaluate one CELL model on the shipped test perturbation — reports all/floor0/floor0.5/>median.
#   M3ED:  SEQ=falcon_indoor_flight_3 bash scripts/eval.sh            # partial_light depth
#   DSEC:  DATASET=dsec SEQ=thun_00_a bash scripts/eval.sh            # sparse depth
# ITERS=24 (default) or 12 (halved-budget variant). CKPT auto-located unless exported.
# Produces a per-frame dump used as the refinement init (scripts/refine.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/scenes.sh
DATASET="${DATASET:-m3ed}"; ITERS="${ITERS:-24}"
: "${SEQ:?set SEQ (test sequence)}"
OD="outputs/eval"; mkdir -p "$OD"

if [ "$DATASET" = "dsec" ]; then
  CKPT="${CKPT:-$(find "$CKPT_ROOT/dsec" -name 'best_model.pth' 2>/dev/null | head -1)}"
  : "${CKPT:?no DSEC checkpoint found under $CKPT_ROOT/dsec (set CKPT=...)}"
  RT="test_set_init_pose/DSEC/test_RT_seq_${SEQ}_5.00_0.50.csv"
  DEPTH=""                                   # DSEC = sparse depth
  DS=dsec
else
  CKPT="${CKPT:-$(find "$CKPT_ROOT/$SEQ" -path '*partial_light*' -name 'best_model.pth' 2>/dev/null | head -1)}"
  : "${CKPT:?no M3ED partial-depth checkpoint under $CKPT_ROOT/$SEQ (set CKPT=...)}"
  RT="test_set_init_pose/M3ED/test_RT_seq_${SEQ}_5.00_0.50.csv"
  DEPTH="--partial_depth partial_light"      # M3ED = partial-light completed depth (train & test)
  DS=m3ed
fi
[ -f "$RT" ] || { echo "missing perturbation file $RT"; exit 1; }
DUMP="$OD/${SEQ}__iter${ITERS}.npz"
echo "=== EVAL $DS $SEQ | iters=$ITERS | ckpt=$CKPT ==="
$PY eval_full.py --ckpt "$CKPT" --tag "${SEQ}_iter${ITERS}" --seq "$SEQ" --dataset "$DS" \
  $DEPTH --test_rt "$RT" --iters "$ITERS" --pweight_floors 0,0.5 --save_poses \
  --dump "$DUMP"
echo "=== eval done -> $DUMP ==="
