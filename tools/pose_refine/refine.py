"""
Edge-based pose refinement core (differentiable, chamfer / distance-transform).

Given 3D edge points X (camera frame) and a target 2D edge map, refine the 6-DOF
camera pose by minimizing the distance-transform cost of the projected points:

    E(xi) = mean_i  huber( DT[ pi( exp(xi) . X_i ) ] )

xi = (omega[3], t[3]) is an se(3) increment applied to X before projection. DT is
bilinearly sampled (fully differentiable) and optimized with LBFGS.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def distance_transform(edge_mask):
    """edge_mask: [H,W] {0,1}. Returns DT [H,W] float = distance (px) to nearest edge."""
    dt = distance_transform_edt(edge_mask < 0.5)
    return torch.from_numpy(dt.astype(np.float32))


def deproject(uv, depth, fx, fy, cx, cy):
    """uv: [N,2] pixel (u,v); depth: [N] metres. Returns X [N,3] in camera frame."""
    z = depth
    x = (uv[:, 0] - cx) * z / fx
    y = (uv[:, 1] - cy) * z / fy
    return torch.stack([x, y, z], dim=1)


def project(X, fx, fy, cx, cy, eps=1e-6):
    z = X[:, 2].clamp(min=eps)
    u = fx * X[:, 0] / z + cx
    v = fy * X[:, 1] / z + cy
    return torch.stack([u, v], dim=1)


def exp_so3(omega):
    theta = torch.linalg.norm(omega) + 1e-12
    k = omega / theta
    K = torch.zeros(3, 3, device=omega.device, dtype=omega.dtype)
    K[0, 1] = -k[2]; K[0, 2] = k[1]
    K[1, 0] = k[2];  K[1, 2] = -k[0]
    K[2, 0] = -k[1]; K[2, 1] = k[0]
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def sample_dt(dt, uv):
    """dt: [H,W]; uv: [N,2] pixels. Bilinear DT value at each uv (border padding)."""
    H, W = dt.shape
    gx = 2 * uv[:, 0] / (W - 1) - 1
    gy = 2 * uv[:, 1] / (H - 1) - 1
    grid = torch.stack([gx, gy], dim=-1)[None, None]           # [1,1,N,2]
    d = F.grid_sample(dt[None, None], grid, align_corners=True, padding_mode="border")
    return d[0, 0, 0]                                          # [N]


def huber(r, delta):
    a = r.abs()
    return torch.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))


def refine_pose(X, dt, calib_crop, init_xi=None, iters=80, huber_delta=3.0, lr=0.3, z_reg=0.0,
                anchor_w=0.0, lock_rot=False, weights=None):
    """
    X          : [N,3] 3D edge points (camera frame, metres)
    dt         : [H,W] distance transform of the target edge map
    calib_crop : (fx, fy, cx, cy) for the cropped image
    init_xi    : [6] initial se(3) (omega, t); default zeros
    anchor_w   : trust-region prior weight on ||xi - init_xi||^2 (keeps the estimate
                 near a trusted init; guards against chamfer local minima)
    weights    : optional [N] per-edge-point weights (e.g. predicted confidence). If given, the chamfer
                 cost is a WEIGHTED mean  sum_i w_i*huber(d_i) / sum_i w_i  (else plain mean). The logged
                 history stays the UNWEIGHTED mean-DT so it's comparable across weighted/uniform runs.
    Returns    : (xi_opt [6], history[list of mean-DT])
    """
    fx, fy, cx, cy = calib_crop
    device = X.device
    anchor = (torch.zeros(6, device=device) if init_xi is None else init_xi.clone().to(device)).detach()
    if weights is not None:
        weights = weights.to(device).float().detach()
        wsum = weights.sum() + 1e-8
    def agg(d):  # weighted or plain mean of huber(d)
        hd = huber(d, huber_delta)
        return (weights * hd).sum() / wsum if weights is not None else hd.mean()
    hist = []

    if lock_rot:
        # Translation-only refinement: rotation held FIXED at the init, optimize ONLY the 3-vector t.
        # This is a genuine reduced (3-DOF) optimization over the translation subspace — not a masked
        # gradient — so rotation is provably unchanged.
        R_fixed = exp_so3(anchor[:3]).detach()
        t = anchor[3:].clone().detach().requires_grad_(True)
        opt = torch.optim.LBFGS([t], lr=lr, max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            Xp = (R_fixed @ X.T).T + t
            uv = project(Xp, fx, fy, cx, cy)
            d = sample_dt(dt, uv)
            cost = agg(d)
            if z_reg > 0:
                cost = cost + z_reg * t[2] ** 2               # regularize weak z DOF (t_z)
            if anchor_w > 0:
                cost = cost + anchor_w * ((t - anchor[3:]) ** 2).sum()
            cost.backward()
            hist.append(float(d.mean()))
            return cost

        opt.step(closure)
        return torch.cat([anchor[:3], t.detach()]), hist

    # Full 6-DOF refinement.
    xi = anchor.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([xi], lr=lr, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        R = exp_so3(xi[:3])
        Xp = (R @ X.T).T + xi[3:]
        uv = project(Xp, fx, fy, cx, cy)
        d = sample_dt(dt, uv)
        cost = agg(d)
        if z_reg > 0:
            cost = cost + z_reg * xi[5] ** 2                  # regularize weak z DOF
        if anchor_w > 0:
            cost = cost + anchor_w * ((xi - anchor) ** 2).sum()
        cost.backward()
        hist.append(float(d.mean()))
        return cost

    opt.step(closure)
    return xi.detach(), hist


def refine_pose_c2f(X, edge_mask, calib_crop, init_xi=None,
                    sigmas=(8.0, 4.0, 2.0, 0.0), **kw):
    """Coarse-to-fine: blur the DT wide first (broad basin -> escapes local minima),
    then progressively sharpen, chaining the estimate. Standard for chamfer alignment."""
    import cv2
    dt0 = distance_transform(edge_mask).numpy()
    xi = torch.zeros(6, device=X.device) if init_xi is None else init_xi
    all_hist = []
    for sg in sigmas:
        dt = dt0 if sg <= 0 else cv2.GaussianBlur(dt0, (0, 0), sigmaX=sg)
        xi, h = refine_pose(X, torch.from_numpy(dt).to(X.device), calib_crop, init_xi=xi, **kw)
        all_hist += h
    return xi, all_hist
