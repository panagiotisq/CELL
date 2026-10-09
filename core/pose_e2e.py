"""Stage 5 — end-to-end pose supervision for LEAR (EPro-PnP).
Builds DIFFERENTIABLE 2D-3D correspondences from the predicted flow so the Monte-Carlo pose loss
backprops into BOTH the flow and the confidence/weight head:
    x3d = deproject(source_pixel, depth)         # perturbed-camera-frame 3D (fixed)
    x2d = [col + flow_x, row + flow_y]           # DIFFERENTIABLE in flow_pred
    w2d = softplus(conf_head)                     # DIFFERENTIABLE in the weight head
    pose_gt = [-T_err, conj(R_err quaternion)]    # verified convention (matches gt_reproj to <0.01px)
Convention (fx,fy,cx,cy) exactly matches core/camera_model.deproject_pytorch + the eval crop shift.
Gated: only used when --pose_e2e is passed; nothing here runs otherwise.
"""
import torch
import torch.nn.functional as F
from core.epropnp import (EProPnP6DoF, LMSolver, RSLMSolver, PerspectiveCamera,
                          AdaptiveHuberPnPCost, MonteCarloPoseLoss)


def intrinsics_from_calib(calib, crop_h, crop_w, crop_x, crop_y):
    """calib: (B,4) = [fx, fy, cx, cy]. Returns fx,fy,cx,cy (B,) with the eval crop principal-point shift.
    Matches eval get_corr: cx += crop_w/2 - (2*crop_y+crop_w)/2 ; cy += crop_h/2 - (2*crop_x+crop_h)/2."""
    fx = calib[:, 0]; fy = calib[:, 1]
    cx = calib[:, 2] + (crop_w / 2. - (2 * crop_y + crop_w) / 2.)
    cy = calib[:, 3] + (crop_h / 2. - (2 * crop_x + crop_h) / 2.)
    return fx, fy, cx, cy


def build_correspondences(flow_pred, depth_norm, calib, crop, max_depth=10.,
                          n_sub=1024, generator=None):
    """flow_pred: (B,2,H,W) full-res [x=col disp, y=row disp]; depth_norm: (B,1,H,W) depth/max_depth.
    Returns per-sample lists x3d (Ni,3) [no grad], x2d (Ni,2) [grad in flow], and K (B,3,3).
    Correspondences = pixels with depth>0 (== valid LiDAR / GT-flow support), subsampled to n_sub."""
    B, _, H, W = flow_pred.shape
    crop_h, crop_w, crop_x, crop_y = crop
    fx, fy, cx, cy = intrinsics_from_calib(calib, crop_h, crop_w, crop_x, crop_y)
    x3d_l, x2d_l = [], []
    K = flow_pred.new_zeros(B, 3, 3)
    K[:, 0, 0] = fx; K[:, 1, 1] = fy; K[:, 0, 2] = cx; K[:, 1, 2] = cy; K[:, 2, 2] = 1.
    for b in range(B):
        d = depth_norm[b, 0]                                   # (H,W) normalized
        mask = d > 0
        idx = mask.nonzero(as_tuple=False)                     # (Ni,2) = (row,col)
        if idx.shape[0] == 0:
            x3d_l.append(None); x2d_l.append(None); continue
        if idx.shape[0] > n_sub:
            sel = torch.randperm(idx.shape[0], device=idx.device, generator=generator)[:n_sub]
            idx = idx[sel]
        r = idx[:, 0].float(); c = idx[:, 1].float()
        z = d[idx[:, 0], idx[:, 1]] * max_depth                # (Ni,) meters, no grad (input)
        X = (c - cx[b]) * z / fx[b]
        Y = (r - cy[b]) * z / fy[b]
        x3d = torch.stack([X, Y, z], dim=-1)                   # (Ni,3)
        fx_disp = flow_pred[b, 0][idx[:, 0], idx[:, 1]]        # (Ni,) grad
        fy_disp = flow_pred[b, 1][idx[:, 0], idx[:, 1]]
        x2d = torch.stack([c + fx_disp, r + fy_disp], dim=-1)  # (Ni,2) grad in flow
        x3d_l.append(x3d); x2d_l.append(x2d)
    return x3d_l, x2d_l, K


def pose_gt_from_err(T_err, R_err):
    """T_err: (B,3); R_err: (B,4) quaternion [w,i,j,k]. Returns pose_gt (B,7)=[-T_err, conj(R_err)]."""
    t = -T_err
    q = torch.stack([R_err[:, 0], -R_err[:, 1], -R_err[:, 2], -R_err[:, 3]], dim=-1)
    return torch.cat([t, q], dim=-1)


class PoseE2ELoss(torch.nn.Module):
    """Monte-Carlo pose loss (EPro-PnP) over differentiable flow correspondences + a weight head.
    loss = MC-KL pose loss + w_reg*(Huber trans + quat rot regression on the differentiable LM pose).
    Averaged over the batch (per-sample num_obj=1; robust to variable #correspondences)."""
    def __init__(self, crop, max_depth=10., n_sub=1024, mc_samples=512, reg_weight=1.0,
                 img_border=200., relative_delta=0.1, beta_pred=1.0, alpha_tgt=1.0):
        super().__init__()
        self.crop = crop; self.max_depth = max_depth; self.n_sub = n_sub
        self.reg_weight = reg_weight; self.img_border = img_border; self.relative_delta = relative_delta
        self.beta_pred = beta_pred  # >1 up-weights L_pred (log-partition/parallax) vs L_tgt (depth-biasing)
        self.alpha_tgt = alpha_tgt  # weight on L_tgt; 0 -> L_pred-only (drop the depth-biasing target term)
        self.rs_num_points = 16
        # frames must have at least this many valid correspondences: the RANSAC init solver samples
        # rs_num_points WITHOUT replacement (torch.multinomial), so N must exceed it with margin.
        self.min_pts = max(32, 2 * self.rs_num_points)
        self.epropnp = EProPnP6DoF(
            mc_samples=mc_samples, num_iter=4,
            solver=LMSolver(dof=6, num_iter=5,
                            init_solver=RSLMSolver(dof=6, num_points=self.rs_num_points, num_proposals=4, num_iter=3)))
        self.mc_loss = MonteCarloPoseLoss()

    def _cam(self, K1):
        crop_h, crop_w = self.crop[0], self.crop[1]
        lb = K1.new_tensor([[-self.img_border, -self.img_border]])
        ub = K1.new_tensor([[crop_w + self.img_border, crop_h + self.img_border]])
        return PerspectiveCamera(cam_mats=K1, z_min=0.1, lb=lb, ub=ub)

    def forward(self, flow_pred, conf_pred, depth_norm, calib, T_err, R_err, generator=None,
                detach_flow=False):
        """flow_pred (B,2,H,W); conf_pred (B,1,H,W) raw logits; depth_norm (B,1,H,W).
        detach_flow=True -> x2d uses flow.detach() so the pose loss trains ONLY the conf head (idea 2:
        decoupled). Weights and correspondences use the SAME sampled source pixels. Returns (loss, metrics)."""
        dev = flow_pred.device
        flow_x2d = flow_pred.detach() if detach_flow else flow_pred
        B, _, H, W = flow_pred.shape
        crop_h, crop_w, crop_x, crop_y = self.crop
        fx, fy, cx, cy = intrinsics_from_calib(calib.to(dev), crop_h, crop_w, crop_x, crop_y)
        pose_gt = pose_gt_from_err(T_err.to(dev), R_err.to(dev))
        # weights are BOUNDED per-frame (softmax over the sampled correspondences x N -> sum=N, mean=1),
        # so the conf head can only REDISTRIBUTE a fixed budget, never grow weights unbounded (which
        # explodes the MC log-partition and lets the flow diverge). Matches EPro-PnP's softmax*scale.
        conf_logits = conf_pred[:, 0]  # (B,H,W) raw logits
        losses, mc_terms, reg_terms, npts = [], [], [], []
        for b in range(B):
            d = depth_norm[b, 0]; mask = d > 0
            idx = mask.nonzero(as_tuple=False)
            if idx.shape[0] < self.min_pts:
                continue
            if idx.shape[0] > self.n_sub:
                sel = torch.randperm(idx.shape[0], device=idx.device, generator=generator)[:self.n_sub]
                idx = idx[sel]
            ri, ci = idx[:, 0], idx[:, 1]
            r = ri.float(); c = ci.float()
            z = d[ri, ci] * self.max_depth
            X = (c - cx[b]) * z / fx[b]; Y = (r - cy[b]) * z / fy[b]
            x3d = torch.stack([X, Y, z], -1)[None]                         # (1,N,3)
            x2d = torch.stack([c + flow_x2d[b, 0][ri, ci], r + flow_x2d[b, 1][ri, ci]], -1)[None]
            w = torch.softmax(conf_logits[b][ri, ci], dim=0) * float(ri.shape[0])  # (N,) sum=N, mean=1
            w2d = torch.stack([w, w], -1)[None]                            # (1,N,2)
            K1 = flow_pred.new_zeros(1, 3, 3)
            K1[0, 0, 0] = fx[b]; K1[0, 1, 1] = fy[b]; K1[0, 0, 2] = cx[b]; K1[0, 1, 2] = cy[b]; K1[0, 2, 2] = 1.
            cost = AdaptiveHuberPnPCost(relative_delta=self.relative_delta); cost.set_param(x2d, w2d)
            _, _, pop, _, logw, ctgt = self.epropnp.monte_carlo_forward(
                x3d, x2d, w2d, self._cam(K1), cost,
                pose_init=pose_gt[b:b + 1], force_init_solve=True, with_pose_opt_plus=True)
            # normalize the MC loss by point-count N so its magnitude is per-correspondence O(1),
            # not O(N) -> keeps lambda meaningful and comparable to the flow anchor across frames.
            lmc = self.mc_loss(logw, ctgt, float(x3d.shape[1]), beta_pred=self.beta_pred, alpha_tgt=self.alpha_tgt)
            pg = pose_gt[b:b + 1]
            lt = (pop[:, :3] - pg[:, :3]).norm(dim=-1); beta = 0.05
            # UNCLAMPED (loose cap only for the rare degenerate solve) so pose regression truly ANCHORS
            # the solved pose to GT -- this is what counteracts the MC log-partition term.
            lt = torch.where(lt < beta, 0.5 * lt.square() / beta, lt - 0.5 * beta).clamp(max=5.).mean()
            dot = (pop[:, None, 3:] @ pg[:, 3:, None]).squeeze(-1).squeeze(-1)
            lr_ = ((1 - dot.square()) * 2).clamp(max=5.).mean()
            loss_b = lmc + self.reg_weight * (lt + lr_)
            if torch.isfinite(loss_b):
                losses.append(loss_b); mc_terms.append(float(lmc)); reg_terms.append(float(lt + lr_)); npts.append(idx.shape[0])
        if not losses:
            z = flow_pred.sum() * 0.0
            return z, {'pose_loss': 0.0, 'pose_mc': 0.0, 'pose_reg': 0.0, 'pose_npts': 0}
        loss = torch.stack(losses).mean()
        m = {'pose_loss': float(loss), 'pose_mc': sum(mc_terms) / len(mc_terms),
             'pose_reg': sum(reg_terms) / len(reg_terms), 'pose_npts': sum(npts) / len(npts)}
        return loss, m
