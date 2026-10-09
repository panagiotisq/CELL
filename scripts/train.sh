#!/usr/bin/env bash
# Train one CELL model — exact paper recipe (scratch, 100 ep, decoupled β1.5 confidence).
#   M3ED (per scene):  SEQ=falcon_indoor_flight_3 bash scripts/train.sh
#   DSEC (one model):  DATASET=dsec bash scripts/train.sh
# TRAIN/EV are looked up from scenes.sh when not exported. Geometry is auto (core/scene_config.py).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/scenes.sh
DATASET="${DATASET:-m3ed}"

RECIPE="--conf_head --pose_e2e --pose_e2e_decouple --pose_e2e_beta_pred 1.5 \
  --conf_feats net,cnet,motion --conf_kernel 1 --conf_nets 4 \
  --pose_e2e_weight 0.1 --pose_e2e_reg 20 --pose_e2e_npts 512 --pose_e2e_warmup 300"
BASE="--backbone edge --use_feature_fusion --gpus 0 --lr 4e-5 --wdecay 0.00005 \
  --epochs 100 --evaluate_interval 5 --save_log"

if [ "$DATASET" = "dsec" ]; then
  # single universal model: sparse depth, all training sequences (no partial-depth completion).
  EV="${EV:-$EV_NIGHT}"           # DSEC rep = ours_pre_100000
  RUN_TAG="scratch100_decplB15"
  echo "=== TRAIN DSEC (single model, sparse) | ev=$EV | tag=$RUN_TAG ==="
  $PY main.py --data_path "$DSEC_DATA" --dataset dsec --ev_input "$EV" \
    --test_sequence "${SEQ:-thun_00_a}" \
    $BASE $RECIPE --run_tag "$RUN_TAG"
else
  : "${SEQ:?set SEQ (M3ED test sequence, e.g. falcon_indoor_flight_3)}"
  TRAIN="${TRAIN:-${M3ED_TRAIN[$SEQ]:-}}"; : "${TRAIN:?no train sequence for $SEQ (set TRAIN=...)}"
  EV="${EV:-${M3ED_EV[$SEQ]:-$EV_HALF}}"
  RUN_TAG="scratch100_decplB15_partial_light"
  echo "=== TRAIN M3ED $SEQ | train=$TRAIN | ev=$EV | partial depth completion | tag=$RUN_TAG ==="
  $PY main.py --data_path "$M3ED_DATA" --dataset m3ed --ev_input "$EV" \
    --test_sequence "$SEQ" --train_sequence "$TRAIN" \
    --partial_depth partial_light --dense_flow \
    $BASE $RECIPE --run_tag "$RUN_TAG"
fi
echo "=== train done: $DATASET ${SEQ:-} (checkpoint under $CKPT_ROOT/) ==="
