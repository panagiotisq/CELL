"""
Single-frame edge extractor for the pose-refinement pipeline.

Adapted from myacezero2/convert_m3ed_pipeline.py `extract_e2egs_edges`:
  - runs the local-variance structural detector on ONE frame (no temporal stack)
  - takes an IWE frame's activity magnitude (|pos|+|neg|) as the single channel
  - NO dilation by default (thin edges -> sharper distance-transform minimum)
  - params scaled for the 360x640 IWE (vs the original full-res 1280x720)
"""
import numpy as np
import cv2


def extract_edges(iwe_2ch, patch=3, sigma=1.0, tau_pct=70, min_area=30, dilate=0):
    """iwe_2ch: [H, W, 2] float IWE (pos/neg counts). Returns [H, W] float {0,1} edge mask."""
    mag = (np.abs(iwe_2ch[..., 0]) + np.abs(iwe_2ch[..., 1])).astype(np.float64)
    m = mag.max()
    if m > 0:
        mag = mag / m * 255.0

    f = cv2.GaussianBlur(mag, (0, 0), sigmaX=sigma)
    mean_f = cv2.blur(f, (patch, patch))
    mean_f_sq = cv2.blur(f ** 2, (patch, patch))
    local_var = np.clip(mean_f_sq - mean_f ** 2, 0, None)   # high at structure/edges

    samp = local_var[::2, ::2]
    tau = np.percentile(samp[samp > 0], tau_pct) if np.any(samp > 0) else 0.0
    tau = max(tau, 1e-5)

    M_bin = np.where(local_var > tau, 255, 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(M_bin, connectivity=8)
    clean = np.zeros_like(M_bin)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        clean = cv2.dilate(clean, k, iterations=dilate)
    return (clean > 0).astype(np.float32)


def skeletonize(mask):
    """Zhang-Suen thinning (pure numpy) -> 1-px medial axis. mask: [H,W] {0,1}.

    Correct way to thin an event ACTIVITY map (which is already a bright edge band):
    thresholding + thinning collapses the band to its centerline, avoiding the
    double edges that Canny would produce on such a map.
    """
    img = (mask > 0).astype(np.uint8).copy()
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = 0

    def _cond(im, step):
        P2 = np.roll(im, 1, 0);  P6 = np.roll(im, -1, 0)
        P4 = np.roll(im, -1, 1); P8 = np.roll(im, 1, 1)
        P3 = np.roll(P2, -1, 1); P9 = np.roll(P2, 1, 1)
        P5 = np.roll(P6, -1, 1); P7 = np.roll(P6, 1, 1)
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
        A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8) for k in range(8))
        c = ((P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)) if step == 0 \
            else ((P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0))
        return (im == 1) & (B >= 2) & (B <= 6) & (A == 1) & c

    while True:
        d1 = _cond(img, 0); img[d1] = 0
        d2 = _cond(img, 1); img[d2] = 0
        if d1.sum() == 0 and d2.sum() == 0:
            break
    return img.astype(np.float32)


def extract_thin_edges(iwe_2ch, **kw):
    """Event thin edges = structure mask -> skeletonized centerline (single-pixel)."""
    return skeletonize(extract_edges(iwe_2ch, dilate=0, **kw))


def geometric_depth_edges(dense_depth, sparse_depth=None, max_depth=10.0,
                          grad_thresh_m=0.3, gate_dilate=5, thin=True):
    """LiDAR thin edges = STRONG depth discontinuities (real object boundaries).

    Uses the depth-gradient magnitude (in metres) thresholded high -> keeps real
    depth steps, drops the smooth densification/interpolation artifacts that Canny
    fired on. Optionally gated to pixels near real LiDAR returns (sparse_depth>0),
    then skeletonized to a 1-px edge.

    dense_depth / sparse_depth: [H,W] normalized depth (0..1 = 0..max_depth m).
    """
    d_m = dense_depth.astype(np.float64) * max_depth
    gx = cv2.Sobel(d_m, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(d_m, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy) / 4.0            # Sobel-3 gain ~4 -> ~metres/px
    edge = (grad > grad_thresh_m).astype(np.uint8)
    if sparse_depth is not None and gate_dilate > 0:
        cov = cv2.dilate((sparse_depth > 0).astype(np.uint8),
                         np.ones((gate_dilate, gate_dilate), np.uint8))
        edge = edge * cov
    edge = edge.astype(np.float32)
    return skeletonize(edge) if thin else edge
