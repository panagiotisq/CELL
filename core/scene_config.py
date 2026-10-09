"""Central per-sequence geometry config for LEAR eval + training — the single source of truth
for crop, calib scaling, and max_depth so every sequence gets the right settings by construction.

DESIGN (safest for a fixed, known sequence set): explicit hardcoded classification (auditable)
+ a runtime resolution assertion (assert_frame_resolution) so any mismatch fails LOUDLY instead
of silently corrupting results.

Two orthogonal axes (verified empirically from the data):
  RESOLUTION  half 360x640 / full 720x1280 / dsec 480x640  -> crop (h,w,x,y) + calib divisor
              (half-res M3ED frames need calib/2; full-res night/fast + DSEC use native calib)
  LOCATION    indoor / outdoor / dsec                       -> max_depth = local-map z-ceiling
              (indoor 10, outdoor 100, DSEC 50 — read directly off local_maps depth, and each
               confirmed by a max_depth sweep peak)

Crop offsets are center-crop derived: x=(H-crop_h)//2, y=(W-crop_w)//2.
"""

# --- explicit classification (hardcoded, auditable) ---------------------------------------
# Full-res M3ED sequences: 720x1280 frames (night + fast). Native calib, 600x960 crop.
_FULL_RES = {
    "car_urban_night_penno_big_loop", "car_urban_night_penno_small_loop",
    "spot_outdoor_night_penno_short_loop",
    "falcon_outdoor_day_fast_flight_1", "falcon_outdoor_day_fast_flight_2",
    "falcon_outdoor_night_high_beams",
    "falcon_outdoor_night_penno_parking_1", "falcon_outdoor_night_penno_parking_2",
}
# Indoor M3ED sequences (half-res, depth ceiling ~10m).
_INDOOR = {
    "falcon_indoor_flight_1", "falcon_indoor_flight_2", "falcon_indoor_flight_3",
    "spot_indoor_building_loop",
}

# --- geometry per class -------------------------------------------------------------------
_GEOM = {
    # class    : crop (h,w,x,y)        calib_div  max_depth  frame_hw (native, for the assertion)
    "indoor":  dict(crop=(288, 512, 36, 64),  calib_div=2.0, max_depth=10.0,  frame_hw=(360, 640)),
    "outdoor": dict(crop=(288, 512, 36, 64),  calib_div=2.0, max_depth=100.0, frame_hw=(360, 640)),
    "full":    dict(crop=(600, 960, 60, 160), calib_div=1.0, max_depth=100.0, frame_hw=(720, 1280)),
    "dsec":    dict(crop=(360, 480, 60, 80),  calib_div=1.0, max_depth=50.0,  frame_hw=(480, 640)),
}


def scene_class(seq, dataset="m3ed"):
    """Return the geometry class for a sequence: indoor / outdoor / full / dsec."""
    if dataset == "dsec":
        return "dsec"
    if seq in _FULL_RES:
        return "full"
    if seq in _INDOOR:
        return "indoor"
    return "outdoor"          # default: M3ED outdoor half-res (the largest, safe-default group)


def scene_geometry(seq, dataset="m3ed"):
    """Full geometry dict for a sequence: crop (h,w,x,y), calib_div, max_depth, frame_hw, klass."""
    k = scene_class(seq, dataset)
    g = dict(_GEOM[k]); g["klass"] = k
    return g


def scene_ev_input(seq, dataset="m3ed"):
    """Default event-representation dir (event_frames_<ev_input>) per sequence.
    Tied to how the data was generated: night full-res = ours_pre_100000; fast_flight full-res =
    ours_denoise_stc_trail_pre_100000 (no _half); indoor+outdoor half-res = ..._half; DSEC = ours_pre_100000."""
    if dataset == "dsec":
        return "ours_pre_100000"
    if seq in ("falcon_outdoor_day_fast_flight_1", "falcon_outdoor_day_fast_flight_2"):
        return "ours_denoise_stc_trail_pre_100000"          # fast: full-res, stc_trail, no _half
    if seq in _FULL_RES:                                     # night: full-res
        return "ours_pre_100000"
    return "ours_denoise_stc_trail_pre_100000_half"         # indoor + outdoor half-res


def assert_frame_resolution(seq, dataset, frame_h, frame_w):
    """Loud failure if a sequence's ACTUAL frame resolution disagrees with its class.
    This is the safety net: a mis-generated or mis-classified sequence errors here instead of
    silently running with the wrong crop/calib. E.g. a half-res night frame would trip this."""
    exp_h, exp_w = _GEOM[scene_class(seq, dataset)]["frame_hw"]
    if (frame_h, frame_w) != (exp_h, exp_w):
        raise ValueError(
            f"[scene_config] '{seq}' (dataset={dataset}, class='{scene_class(seq, dataset)}'): "
            f"frame is {frame_h}x{frame_w} but class expects {exp_h}x{exp_w}. "
            f"Fix the classification in core/scene_config.py (e.g. move seq between _FULL_RES / "
            f"_INDOOR / outdoor) so crop+calib match its actual resolution."
        )
