import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from quaternion_distances import quaternion_distance
import numpy as np
import visibility


def sequence_loss_reweight(flow_preds, flow_gt, conf_w, gamma=0.8, MAX_FLOW=400, eps=1e-5):
    """STAGE 5 idea-2 (decoupled): flow loss REWEIGHTED per-pixel by a DETACHED confidence weight,
    so flow training focuses on the (pose-supervised, depth-unbiased) high-confidence regions.
    conf_w: (B,1,H,W) positive weight (already detached). Weighted-MEAN over valid pixels (magnitude
    preserved, so it stays a well-scaled flow loss). Same gamma iter-weighting as sequence_loss."""
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = ((flow_gt[:, 0] != 0) | (flow_gt[:, 1] != 0)) & (mag < MAX_FLOW)   # (B,H,W)
    vf = valid.float()
    w = (conf_w[:, 0] * vf)                                                    # (B,H,W) weight on valid pixels
    wsum = w.sum(dim=[1, 2]).clamp(min=eps)                                    # (B,)
    n = len(flow_preds)
    flow_loss = 0.0
    for i in range(n):
        iw = gamma ** (n - i - 1)
        perpix = torch.norm(flow_preds[i] - flow_gt, dim=1)                    # (B,H,W)
        L = (w * perpix).sum(dim=[1, 2]) / wsum                                # weighted mean per image
        flow_loss += iw * L.mean()
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt().view(-1)[valid.view(-1)]
    return flow_loss, {'epe': epe.mean().item()}


def sequence_loss_zero(flow_preds, valid_mask, gamma=0.8, conf_w=None):
    """ZERO-FLOW branch (I2D-LocX Eq.15): the network is run a 2nd time on the GT-pose (aligned)
    depth, where the true displacement is 0 everywhere the depth projects. Same masked-EPE form as
    the main loss, but target = 0 and the validity mask is GIVEN explicitly (the aligned-depth
    projection, ~15-20%) -- it CANNOT be derived from flow_gt!=0 the way sequence_loss does, because
    here the target flow is identically zero. Supervising ||flow_i - 0|| pushes the correlation to
    peak at displacement 0 => co-located depth/event features must match.
    flow_preds: list of (B,2,H,W).  valid_mask: (B,H,W) bool where the aligned depth is valid.
    conf_w: optional (B,1,H,W) DETACHED per-pixel weight (variant 1b = softplus(conf) from the zero
    pass, applied EXACTLY like the decoupled main-loss reweighting -> weighted mean over valid pixels).
    NOTE epe_zero (the logged signal) stays UNWEIGHTED so it's comparable across uniform/conf variants."""
    vf = valid_mask.float()                                          # (B,H,W)
    if conf_w is not None:
        vf = vf * conf_w[:, 0]                                       # weight valid pixels by detached conf
    vsum = vf.sum(dim=[1, 2]).clamp(min=1e-5)                        # (B,)
    n = len(flow_preds)
    flow_loss = 0.0
    for i in range(n):
        iw = gamma ** (n - i - 1)
        perpix = torch.norm(flow_preds[i], dim=1)                    # ||flow_i - 0|| , (B,H,W)
        L = (vf * perpix).sum(dim=[1, 2]) / vsum                     # masked mean per image
        flow_loss += iw * L.mean()
    with torch.no_grad():
        m = valid_mask.view(-1)
        epe0 = torch.norm(flow_preds[-1], dim=1).view(-1)[m]
        epe0 = epe0.mean().item() if epe0.numel() else 0.0
    return flow_loss, {'epe_zero': epe0}


def build_flow_target(flow_gt, valid_full):
    """VERSION B helper: map each 1/8 source cell to its GT-flow target cell on the 1/8 event grid.

    flow_gt : (B,2,H,W) full-res GT flow, ch0=x/col ch1=y/row, so (perturbed pixel p)+flow_gt(p) =
              GT-aligned (event) pixel. SPARSE (nonzero only at valid perturbed-depth pixels).
    valid_full: (B,1,H,W) bool where flow_gt is defined (perturbed-depth valid).
    Returns (B,h,w) LONG flat target index (tr*w+tc) into the 1/8 event grid. Uses MASK-AWARE pooling
    (avg only over valid subpixels) so sparse zeros don't bleed in (bilinear would corrupt it).
    """
    vf = valid_full.float()
    cnt = F.avg_pool2d(vf, 8, 8).clamp(min=1e-6)                       # (B,1,h,w)
    fx = (F.avg_pool2d(flow_gt[:, 0:1] * vf, 8, 8) / cnt)[:, 0]        # (B,h,w) mean x-flow (full-res px)
    fy = (F.avg_pool2d(flow_gt[:, 1:2] * vf, 8, 8) / cnt)[:, 0]        # (B,h,w) mean y-flow
    B, h, w = fx.shape
    dev = fx.device
    cc = torch.arange(w, device=dev).view(1, 1, w).float()            # source col index (1/8 grid)
    rr = torch.arange(h, device=dev).view(1, h, 1).float()            # source row index
    tc = torch.round(cc + fx / 8.0).clamp(0, w - 1).long()           # target col: (8c+flow_x)/8 = c+flow_x/8
    tr = torch.round(rr + fy / 8.0).clamp(0, h - 1).long()
    return tr * w + tc                                                # (B,h,w) long flat index


def correlation_matching_loss(fmap1, fmap2, rows_mask, cols_mask,
                              normalize=False, temperature=0.1, row_weight=None,
                              target_index=None, soft_sigma=0.0):
    """OPTION 3 (correlation-level cross-modal matching, GRU-free).

    fmap1 = F_D (depth/lidar features), fmap2 = F_EV (event features); both (B,C,h,w) at 1/8 res,
    the RAW fnet outputs that CorrBlock consumes. Builds the full correlation S[i,j] = <F_D(i),F_EV(j)>
    and applies a per-row softmax cross-entropy pushing the correct target column to be the row max
    (LoFTR/GMFlowNet-style matching loss). One hop to both encoders, NO GRU.

    target_index: None  -> VERSION A: positive = identity column (aligned-depth input, q=p).
                  (B,h,w) LONG -> VERSION B: positive = the GT-flow-shifted target cell q=p+flow_gt(p)
                  on the REAL (perturbed-input) correlation (use build_flow_target). This is the
                  faithful "make features alike at the known correspondence" on the volume the GRU reads.
    normalize : L2-normalize features (cosine + temperature). For VERSION B use True + temperature~0.1
                (LoFTR) so the softmax over ~thousands of candidates is peaky/learnable; raw dot /sqrt(C)
                is near-flat (Version-A plateaued at uniform = its likely defect).
    soft_sigma: >0 -> soft Gaussian label (std in 1/8 cells) around the target instead of one-hot CE
                (absorbs sub-pixel/rounding + 1-vs-thousands imbalance; stereo soft-CE evidence). 0 = hard.
    rows_mask : (B,h,w) bool source cells to supervise. cols_mask: (B,h,w) bool candidate event cells.
    row_weight: optional (B,h,w) DETACHED weight (e.g. softplus(conf)) for a weighted mean.
    Returns (loss, {'corr_ce','corr_rank1'}); corr_rank1 = fraction of rows whose argmax == target.
    """
    B, C, h, w = fmap1.shape
    HW = h * w
    scale = (1.0 / temperature) if normalize else (1.0 / (float(C) ** 0.5))
    total = 0.0
    nb = 0
    rank1 = 0.0
    for b in range(B):
        A = fmap1[b].reshape(C, HW).t()                     # (HW, C) rows = depth pixels
        Bm = fmap2[b].reshape(C, HW)                        # (C, HW) cols = event pixels
        if normalize:
            A = F.normalize(A, dim=1)
            Bm = F.normalize(Bm, dim=0)
        rows = rows_mask[b].reshape(-1).nonzero(as_tuple=False).squeeze(1)
        cols = cols_mask[b].reshape(-1).nonzero(as_tuple=False).squeeze(1)
        if rows.numel() < 1 or cols.numel() < 2:
            continue
        # target flat index per source row: identity (A) or flow-shifted (B)
        tgt = rows if target_index is None else target_index[b].reshape(-1)[rows]
        inv = torch.full((HW,), -1, dtype=torch.long, device=A.device)
        inv[cols] = torch.arange(cols.numel(), device=A.device)
        pos = inv[tgt]                                      # target column position within `cols`
        keep = pos >= 0                                     # drop rows whose target isn't event-active
        rows, pos, tgt = rows[keep], pos[keep], tgt[keep]
        if rows.numel() < 1:
            continue
        S = (A[rows] @ Bm[:, cols]) * scale                 # (Nv, Ne) logits
        if soft_sigma > 0:
            # soft Gaussian label over cols by spatial distance to the target cell
            col_r = (cols // w).float(); col_c = (cols % w).float()          # (Ne,)
            tr = (tgt // w).float()[:, None]; tc = (tgt % w).float()[:, None]  # (Nv,1)
            d2 = (col_r[None, :] - tr) ** 2 + (col_c[None, :] - tc) ** 2      # (Nv, Ne)
            lt = torch.softmax(-d2 / (2.0 * soft_sigma * soft_sigma), dim=1)  # normalized soft target
            ce = -(lt * torch.log_softmax(S, dim=1)).sum(1)                   # (Nv,)
        else:
            ce = F.cross_entropy(S, pos, reduction='none')  # (Nv,)
        if row_weight is not None:
            rw = row_weight[b].reshape(-1)[rows]            # (Nv,) detached conf weight per row
            total = total + (ce * rw).sum() / rw.sum().clamp(min=1e-5)
        else:
            total = total + ce.mean()
        nb += 1
        with torch.no_grad():
            rank1 += (S.argmax(1) == pos).float().mean().item()
    if nb == 0:
        return fmap1.sum() * 0.0, {'corr_ce': 0.0, 'corr_rank1': 0.0}
    loss = total / nb
    return loss, {'corr_ce': loss.item(), 'corr_rank1': rank1 / nb}


def sequence_loss(flow_preds, flow_gt, gamma=0.8, MAX_FLOW=400):
    """ Loss function defined over sequence of flow predictions """

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    Mask = torch.zeros([flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2],
                        flow_gt.shape[3]]).to(flow_gt.device)
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    valid = mask & (mag < MAX_FLOW)
    Mask[:, 0, :, :] = valid
    Mask[:, 1, :, :] = valid
    Mask = Mask != 0
    mask_sum = torch.sum(mask, dim=[1, 2])

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        Loss_reg = (flow_preds[i] - flow_gt) * Mask
        Loss_reg = torch.norm(Loss_reg, dim=1)
        Loss_reg = torch.sum(Loss_reg, dim=[1, 2])
        Loss_reg = Loss_reg / (mask_sum + 1e-5)
        flow_loss += i_weight * Loss_reg.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
    }

    return flow_loss, metrics


def sequence_loss_attn(flow_preds, flow_gt, conf, conf_alpha=10.0, conf_scale=5.0,
                       gamma=0.8, MAX_FLOW=400):
    """EGFS-style ATTENUATED sequence loss. Earlier RAFT iters keep plain masked-L2 supervision
    (so the flow still trains / converges). The FINAL iteration's per-pixel error is weighted by
    the confidence: c*tanh(err/scale) - alpha*log c on valid pixels. conf must be NON-detached and
    err carries grad, so confidence co-adapts with the flow (network gives up where c is low)."""
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    valid = (mask & (mag < MAX_FLOW))                       # [B,H,W]
    vf = valid.float()
    mask_sum = torch.sum(valid, dim=[1, 2]).clamp(min=1)    # [B]
    c = conf[:, 0, :, :]                                    # [B,H,W] (softplus'd, >0)
    n = len(flow_preds)
    flow_loss = 0.0
    for i in range(n):
        w = gamma ** (n - i - 1)
        perpix = torch.norm(flow_preds[i] - flow_gt, dim=1)   # [B,H,W]
        if i < n - 1:
            L = (perpix * vf).sum(dim=[1, 2]) / mask_sum
        else:
            r_hat = torch.tanh(perpix / conf_scale)
            term = (c * r_hat - conf_alpha * torch.log(c)) * vf
            L = term.sum(dim=[1, 2]) / mask_sum
        flow_loss += w * L.mean()
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt().view(-1)[valid.view(-1)]
    metrics = {'epe': epe.mean().item(), 'conf_mean': c[valid].mean().item()}
    return flow_loss, metrics


def sequence_loss_plus(flow_preds, flow_gt, flow_ref, gamma=0.8, MAX_FLOW=400):
    """ Loss function defined over sequence of flow predictions """

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    Mask = torch.zeros([flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2],
                        flow_gt.shape[3]]).to(flow_gt.device)
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    valid = mask & (mag < MAX_FLOW)
    Mask[:, 0, :, :] = valid
    Mask[:, 1, :, :] = valid
    Mask = Mask != 0

    flow_gt_plus = flow_ref.clone()
    flow_gt_plus[Mask] = flow_gt[Mask]

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        Loss_reg = flow_preds[i] - flow_gt_plus
        Loss_reg = torch.norm(Loss_reg, dim=1)
        Loss_reg = torch.sum(Loss_reg, dim=[1, 2])
        Loss_reg = Loss_reg / (flow_gt.shape[2] * flow_gt.shape[3])
        flow_loss += i_weight * Loss_reg.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
    }

    return flow_loss, metrics


def DepthReconLoss(source_depth_maps, target_depth_map, gamma=0.8):
    n_predictions = len(source_depth_maps)
    loss = 0.0
    # target_depth_map = target_depth_map * 2 - 1
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        source_depth_map = source_depth_maps[i]
        mask = source_depth_map > 0

        loss_n = torch.sum(torch.abs((source_depth_map - target_depth_map) * mask), dim=[1,2,3]) / (torch.sum(mask, dim=[1,2,3]) + 1.)
        loss += i_weight * torch.mean(loss_n)
    
    return loss


def ChamferLossOneWay2D(source_depth_map, depth_mask, flow_preds, target_event_frame, gamma=0.8):
    mask = depth_mask>0.1
    source_depth_map = source_depth_map * mask

    n_predictions = len(flow_preds)
    loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        source_depth_map = warp(source_depth_map, -1 * flow_preds[i])

        B, _, H, W = source_depth_map.shape

        # Get indices of valid depth points in the source depth map
        source_mask = source_depth_map > 0  # Non-zero points
        source_points = torch.nonzero(source_mask, as_tuple=False)  # [N_source, 3]

        # Get indices of valid depth points in the target depth map
        target_mask = (target_event_frame[:, 0, :, :] > 0) + (target_event_frame[:, 1, :, :] > 0)  # Non-zero points
        target_mask = target_mask.unsqueeze(1)
        target_points = torch.nonzero(target_mask, as_tuple=False)  # [N_target, 3]

        if source_points.shape[0] == 0 or target_points.shape[0] == 0:
            # If there are no valid points in either depth map, return zero loss
            return torch.tensor(0.0, device=source_depth_map.device)

        # Extract x, y coordinates of source and target points
        source_coords = source_points[:, 1:].float()  # Ignore batch index
        target_coords = target_points[:, 1:].float()  # Ignore batch index

        if target_coords.shape[0] < source_coords.shape[0]:
            if target_coords.shape[0] < 30000:
                indices = torch.randperm(source_coords.shape[0])[:target_coords.shape[0]]
                source_coords = source_coords[indices, :]
            else:
                indices = torch.randperm(source_coords.shape[0])[:30000]
                source_coords = source_coords[indices, :]
                indices = torch.randperm(target_coords.shape[0])[:30000]
                target_coords = target_coords[indices, :]
        else:
            if source_coords.shape[0] < 30000:
                indices = torch.randperm(target_coords.shape[0])[:source_coords.shape[0]]
                target_coords = target_coords[indices, :]
            else:
                indices = torch.randperm(source_coords.shape[0])[:30000]
                source_coords = source_coords[indices, :]
                indices = torch.randperm(target_coords.shape[0])[:30000]
                target_coords = target_coords[indices, :]  

        # Compute pairwise distances between each source point and each target point
        dist_matrix = torch.cdist(source_coords, target_coords, p=2)  # [N_source, N_target]

        # Get the minimum distance for each source point to the nearest target point
        min_dist, _ = torch.min(dist_matrix, dim=1)  # [N_source]

        # One-way Chamfer loss (sum of distances or mean depending on preference)
        chamfer_loss = torch.mean(min_dist)

        loss += i_weight * chamfer_loss.cuda()


    return loss


def ConsistencyLoss(source_depth_map, depth_mask, flow_preds, target_event_frame, gamma=0.8):
    mask = depth_mask!=0
    source_depth_map = source_depth_map * mask

    n_predictions = len(flow_preds)
    loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        source_depth_map = warp(source_depth_map, -1 * flow_preds[i])

        source_mask = source_depth_map != 0  # Non-zero points
        source_mask = source_mask.float()

        target_mask = (target_event_frame[:, 0, :, :] != 0) + (target_event_frame[:, 1, :, :] != 0)  # Non-zero points
        target_mask = target_mask.unsqueeze(1).float()

        consist_loss = (source_mask - target_mask) ** 2
        consist_loss = torch.sum(consist_loss, dim=[1, 2, 3])

        consist_loss = consist_loss / (source_mask.sum() + 1e-5)

        loss += i_weight * consist_loss.mean()


    return loss


def ClassifyLoss(depth_mask, ground_truth, flow=None, mask=None, loss_func="Cross_Entropy_Loss"):
    if loss_func == "Cross_Entropy_Loss":
        # cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        depth_mask = torch.cat((1.-depth_mask, depth_mask), dim=1)
        loss = criterion(depth_mask, ground_truth)
    elif loss_func == "Focal_Loss":
        # focal loss
        alpha=0.25
        gamma=2.0
        criterion = nn.CrossEntropyLoss(reduction='none')
        depth_mask = torch.cat((1.-depth_mask, depth_mask), dim=1)
        ce_loss = criterion(depth_mask, ground_truth)
        pt = torch.exp(-ce_loss)
        loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
    elif loss_func == "Weighted_Cross_Entropy_Loss":
        # weighted cross-entropy loss
        depth_mask = torch.cat((1.-depth_mask, depth_mask), dim=1)
        count_neg = (ground_truth == 0).sum()
        count_pos = (ground_truth == 1).sum()
        beta = count_neg / (count_neg + count_pos)
        pos_weight = beta / (1 - beta)
        weights = torch.tensor([beta, pos_weight], device=ground_truth.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        loss = criterion(depth_mask, ground_truth)
    elif loss_func == "Sequence_Weighted_Cross_Entropy_Loss":
        # weighted cross-entropy loss for a sequence of predictions
        count_neg = (ground_truth == 0).sum()
        count_pos = (ground_truth == 1).sum()
        beta = count_neg / (count_neg + count_pos)
        pos_weight = beta / (1 - beta)
        weights = torch.tensor([beta, pos_weight], device=ground_truth.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        n_predictions = len(depth_mask)
        gamma=0.8
        loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            depth_mask_sub = depth_mask[i]
            depth_mask_sub = torch.cat((1.-depth_mask_sub, depth_mask_sub), dim=1)
            loss_i = criterion(depth_mask_sub, ground_truth)
            loss += i_weight * loss_i
        return loss, loss_i
    else:
        raise "Loss Function doesn't exist"
    return loss


class SigLoss(nn.Module):
    """SigLoss.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=1.,
                 warm_up=False,
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.001 # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""
        
        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth
    



def FuseConsistLoss(edge_gt, edge_pre, depth_gt, depth_pre, flow_pre, flow_gt):
    loss = 0.
    flow_mask = ((flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)).unsqueeze(1)
    loss_edge = ClassifyLoss(edge_pre, edge_gt, flow=flow_pre, mask=flow_mask, loss_func="Masked_Inverse_Warped_Weighted_Cross_Entropy_Loss")
    loss_depth_fn = SigLoss()
    warp_depth_gt = warp(depth_gt, flow_pre)
    warp_depth_gt *= flow_mask
    depth_pre *= flow_mask
    loss_depth = loss_depth_fn(depth_pre, warp_depth_gt)

    return loss_edge + loss_depth



def FeatureTransferLoss(feature, feature_guide):

    return F.mse_loss(feature, feature_guide)



class ProposedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(ProposedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        return total_loss


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask

if __name__ == "__main__":
    flow_preds = torch.tensor([[[[2., 1., 1., 1.],
                                 [4., 1., 1., 1.],
                                 [6., 1., 1., 1.]],

                                [[1., 1., 1., 1.],
                                 [1., 1., 1., 1.],
                                 [1., 1., 1., 1.]]]])
    uncertainty_maps = torch.tensor([[[[1., 1., 1., 1.], 
                                       [1., 1., 1., 1.],
                                       [1., 1., 1., 1.]]]])
    flow_gt = torch.tensor([[[[1., 1., 1., 1.],
                              [1., 1., 1., 1.],
                              [1., 1., 1., 1.]],

                             [[1., 1., 1., 1.],
                              [1., 1., 1., 1.],
                              [1., 1., 1., 1.]]]])