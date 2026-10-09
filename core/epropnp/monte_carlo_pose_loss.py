"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import torch
import torch.nn as nn


class MonteCarloPoseLoss(nn.Module):

    def __init__(self, init_norm_factor=1.0, momentum=0.01):
        super(MonteCarloPoseLoss, self).__init__()
        self.register_buffer('norm_factor', torch.tensor(init_norm_factor, dtype=torch.float))
        self.momentum = momentum

    def forward(self, pose_sample_logweights, cost_target, norm_factor, beta_pred=1.0, alpha_tgt=1.0):
        """
        Args:
            pose_sample_logweights: Shape (mc_samples, num_obj)
            cost_target: Shape (num_obj, )
            norm_factor: Shape ()
            beta_pred: relative weight on the log-partition (L_pred) term. >1 up-weights the
                parallax/geometry-constraining term vs the (depth-biasing) target term L_tgt.
                Default 1.0 == original EPro-PnP KL loss.
            alpha_tgt: weight on the target term L_tgt (the depth-biasing reproj-at-GT cost).
                Set 0.0 to drop L_tgt entirely (L_pred-only). Default 1.0 == original.
        """
        if self.training:
            with torch.no_grad():
                self.norm_factor.mul_(
                    1 - self.momentum).add_(self.momentum * norm_factor)

        loss_tgt = cost_target
        loss_pred = torch.logsumexp(pose_sample_logweights, dim=0)  # (num_obj, )

        loss_pose = alpha_tgt * loss_tgt + beta_pred * loss_pred  # alpha_tgt=0 -> L_pred-only
        loss_pose = torch.where(torch.isnan(loss_pose), torch.zeros_like(loss_pose), loss_pose)  # avoid in-place idx
        # divide by a CLONE: the in-place running-stat update above must not corrupt the saved graph
        # when this loss is called once per sample within a batch before a single backward().
        loss_pose = loss_pose.mean() / self.norm_factor.detach().clone()

        return loss_pose.mean()
