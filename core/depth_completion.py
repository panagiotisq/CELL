import os
import numpy as np
import cv2

# Deployed completion = 'partial_light' (this IS the paper's "partial depth completion").
# 'partial' and 'partial_dc' are DISABLED exploratory variants — kept here for reference but
# rejected in sparse_to_dense() so they can never be selected by accident.
_PARTIAL = ('partial_light',)
# _PARTIAL_DISABLED = ('partial', 'partial_dc')

def sparse_to_dense(sparse, max_depth=100., fill_level='full', dc_tau=None):
    """Morphological depth completion.

    Every level finishes with an unconditional median(5) + bilateral(5) smoothing (the 'blur' step).

    fill_level='full' (DEFAULT) — ORIGINAL behaviour, byte-identical to before. Do not change.
    fill_level='partial_light' — DEPLOYED; this is the paper's "partial depth completion":
        invert -> ONE diamond-3 (3x3 cross) dilation -> median+bilateral smooth -> invert back.
        NO morphological close, NO large-gap (7x7) fill. Raises valid depth from ~20% to ~68%.

    DISABLED exploratory variants (rejected below; kept commented for reference only):
    #  fill_level='partial'    — like 'partial_light' PLUS a 3x3 morphological close (fills more).
    #  fill_level='partial_dc' — like 'partial', then ERASE any filled pixel whose local window of
    #      ORIGINAL real depths spans > dc_tau (a depth discontinuity); dc_tau default 0.15*max_depth.
    """
    # 'partial' / 'partial_dc' were exploratory only and are NOT used in the paper. Disabled so they
    # cannot be selected by accident — use 'partial_light' (the deployed partial depth completion).
    if fill_level in ('partial', 'partial_dc'):
        raise ValueError(f"fill_level={fill_level!r} is a disabled exploratory variant; "
                         f"use 'partial_light' (the paper's partial depth completion).")
    # [disabled: 'partial_dc'] capture original (meters) before inversion for the depth-consistency gate:
    # if fill_level == 'partial_dc':
    #     orig = sparse.copy()
    #     valid0 = orig > 0.1

    ## invert
    valid = sparse > 0.1
    sparse[valid] = max_depth - sparse[valid]

    ## dilate
    if fill_level in _PARTIAL:
        custom_kernel = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=np.uint8)                 # diamond 3x3 — shorter reach, fills only tight gaps
    else:
        custom_kernel = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8)
    sparse = cv2.dilate(sparse, custom_kernel)

    ## close — skipped entirely for the lightest variant
    if fill_level != 'partial_light':
        _close = 3 if fill_level in _PARTIAL else 5
        custom_kernel = np.ones((_close, _close), np.uint8)
        sparse = cv2.morphologyEx(sparse, cv2.MORPH_CLOSE, custom_kernel)

    ## fill (large-gap bridge) — the margin-hallucination step; skipped for every partial variant
    if fill_level not in _PARTIAL:
        invalid = sparse < 0.1
        custom_kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(sparse, custom_kernel)
        sparse[invalid] = dilated[invalid]

    ## blur
    sparse = cv2.medianBlur(sparse, 5)
    sparse = cv2.bilateralFilter(sparse, 5, 1.5, 2.0)

    ## invert
    valid = sparse > 0.1
    sparse[valid] = max_depth - sparse[valid]

    ## [disabled: 'partial_dc'] depth-consistency gate (idea 2): drop fills that straddle a discontinuity
    # if fill_level == 'partial_dc':
    #     tau = dc_tau if dc_tau is not None else 0.15 * max_depth
    #     k = np.ones((7, 7), np.uint8)
    #     local_max = cv2.dilate(orig, k)                       # far neighbour (invalid=0 never raises max)
    #     tmp = orig.copy(); tmp[~valid0] = max_depth * 10.0     # invalid -> big so it never wins the min
    #     local_min = cv2.erode(tmp, k)                          # nearest valid neighbour
    #     has_valid = cv2.dilate(valid0.astype(np.float32), k) > 0
    #     local_range = np.where(has_valid, local_max - local_min, 0.0)
    #     newly_filled = (sparse > 0.1) & (~valid0)              # completed pixels (not original points)
    #     sparse[newly_filled & (local_range > tau)] = 0.0       # straddles an edge -> leave empty

    return sparse