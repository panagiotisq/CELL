#!/usr/bin/env bash
# End-to-end automation: reproduce the paper's per-sequence pipeline (train -> eval -> refine)
# with the exact recipe, for every scene. Long-running; intended to be launched per scene (e.g.
# wrapped in SLURM: `sbatch --export=ALL,SEQ=<seq> ...`) or run sequentially here.
#
#   bash scripts/run_all.sh                       # M3ED (10 scenes) + DSEC, stages train/eval/refine, iters 24 & 12
#   STAGES="eval refine" bash scripts/run_all.sh  # skip training (use existing checkpoints)
#   SCENES="falcon_indoor_flight_3" bash scripts/run_all.sh
#   ITERS_LIST="24" DATASETS="m3ed" bash scripts/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/scenes.sh
STAGES="${STAGES:-train eval refine}"
ITERS_LIST="${ITERS_LIST:-24 12}"
DATASETS="${DATASETS:-m3ed dsec}"
SCENES="${SCENES:-}"                 # empty => all M3ED scenes (keys of M3ED_TRAIN)

run_one() {  # $1=dataset  $2=seq
  local ds=$1 seq=$2
  if [[ " $STAGES " == *" train "* ]]; then DATASET=$ds SEQ=$seq bash scripts/train.sh; fi
  for it in $ITERS_LIST; do
    if [[ " $STAGES " == *" eval "* ]];   then DATASET=$ds SEQ=$seq ITERS=$it bash scripts/eval.sh;   fi
    if [[ " $STAGES " == *" refine "* ]]; then DATASET=$ds SEQ=$seq ITERS=$it bash scripts/refine.sh; fi
  done
}

for ds in $DATASETS; do
  if [ "$ds" = "m3ed" ]; then
    local_scenes="${SCENES:-${!M3ED_TRAIN[*]}}"
    for seq in $local_scenes; do echo "######## M3ED $seq ########"; run_one m3ed "$seq"; done
  elif [ "$ds" = "dsec" ]; then
    # one universal model trained once, then eval/refine each test sequence
    if [[ " $STAGES " == *" train "* ]]; then DATASET=dsec bash scripts/train.sh; fi
    for seq in "${DSEC_TEST[@]}"; do
      echo "######## DSEC $seq ########"
      for it in $ITERS_LIST; do
        if [[ " $STAGES " == *" eval "* ]];   then DATASET=dsec SEQ=$seq ITERS=$it bash scripts/eval.sh;   fi
        if [[ " $STAGES " == *" refine "* ]]; then DATASET=dsec SEQ=$seq ITERS=$it bash scripts/refine.sh; fi
      done
    done
  fi
done
echo "=== run_all complete ==="
