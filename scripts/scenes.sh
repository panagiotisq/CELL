#!/usr/bin/env bash
# CELL scene registry — the exact per-sequence splits + event representation used in the paper.
# Sourced by train.sh / eval.sh / refine.sh / run_all.sh. Edit DATA/CKPT/PY roots via env if needed.
#
# Geometry (crop / calibration / max_depth) is NOT set here — it is resolved automatically per
# sequence by core/scene_config.py for both training and evaluation. Only the test sequence, its
# training sequence(s), and the event representation differ per scene.

# ---- environment (override by exporting before calling) ----
: "${PY:=python}"                                  # python with the CELL env
: "${M3ED_DATA:=data/m3ed}"                         # M3ED data root (per-seq dirs)
: "${DSEC_DATA:=data/dsec_generated}"               # DSEC data root
: "${CKPT_ROOT:=checkpoints}"                       # where trained models are written/read
: "${EV_HALF:=ours_denoise_stc_trail_pre_100000_half}"   # day/indoor half-res STC/Trail rep
: "${EV_NIGHT:=ours_pre_100000}"                    # the two night scenes use this rep (full-res)

# ---- M3ED: 10 per-scene models. test_sequence -> "train_sequence(s)" ----
# Multi-sequence scenes: one sequence held out for test, siblings for train.
# Single-sequence scenes (temporal 70/30 split; test = last 30%): train_sequence = the same
#   sequence name — core/datasets_m3ed.py hardcodes these for the temporal split.
declare -A M3ED_TRAIN=(
  [car_forest_into_ponds_short]="car_forest_into_ponds_long"
  [car_urban_day_penno_small_loop]="car_urban_day_penno_big_loop"
  [car_urban_night_penno_small_loop]="car_urban_night_penno_big_loop"
  [falcon_forest_into_forest_1]="falcon_forest_into_forest_2,falcon_forest_into_forest_4"
  [falcon_indoor_flight_3]="falcon_indoor_flight_1,falcon_indoor_flight_2"
  [falcon_outdoor_day_penno_parking_2]="falcon_outdoor_day_penno_parking_1"
  [spot_forest_road_3]="spot_forest_road_1"
  [spot_indoor_building_loop]="spot_indoor_building_loop"                 # temporal split
  [spot_outdoor_day_penno_short_loop]="spot_outdoor_day_penno_short_loop" # temporal split
  [spot_outdoor_night_penno_short_loop]="spot_outdoor_night_penno_short_loop" # temporal split
)
# event representation per scene (only the two night scenes differ)
declare -A M3ED_EV
for s in "${!M3ED_TRAIN[@]}"; do M3ED_EV[$s]="$EV_HALF"; done
M3ED_EV[car_urban_night_penno_small_loop]="$EV_NIGHT"
M3ED_EV[spot_outdoor_night_penno_short_loop]="$EV_NIGHT"

# the 9 scenes in the paper's main table (car_urban_night is reported only in the supplement)
M3ED_PAPER9=(car_forest_into_ponds_short car_urban_day_penno_small_loop falcon_outdoor_day_penno_parking_2 \
  falcon_indoor_flight_3 spot_indoor_building_loop spot_forest_road_3 falcon_forest_into_forest_1 \
  spot_outdoor_day_penno_short_loop spot_outdoor_night_penno_short_loop)

# ---- DSEC: ONE universal model, sparse depth, ev=ours_pre_100000; evaluated on 13 test sequences ----
DSEC_TEST=(thun_00_a zurich_city_00_b zurich_city_01_f zurich_city_02_e zurich_city_03_a zurich_city_04_f \
  zurich_city_05_b zurich_city_06_a zurich_city_07_a zurich_city_08_a zurich_city_09_e zurich_city_10_b zurich_city_11_c)
