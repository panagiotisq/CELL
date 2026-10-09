"""Vendored EPro-PnP layer (from tjiiv-cprg/EPro-PnP-v2, EPro-PnP-6DoF_v2/lib, MIT license).
Patched for torch>=2.11: torch.solve -> torch.linalg.solve, torch.cholesky -> torch.linalg.cholesky.
Training-only differentiable probabilistic PnP; inference in LEAR keeps poselib unchanged.
"""
from .epropnp import EProPnP6DoF, EProPnP4DoF, EProPnPBase
from .levenberg_marquardt import LMSolver, RSLMSolver
from .camera import PerspectiveCamera
from .cost_fun import HuberPnPCost, AdaptiveHuberPnPCost
from .monte_carlo_pose_loss import MonteCarloPoseLoss
from .common import evaluate_pnp, quaternion_to_rot_mat

__all__ = ['EProPnP6DoF', 'EProPnP4DoF', 'EProPnPBase', 'LMSolver', 'RSLMSolver',
           'PerspectiveCamera', 'HuberPnPCost', 'AdaptiveHuberPnPCost',
           'MonteCarloPoseLoss', 'evaluate_pnp', 'quaternion_to_rot_mat']
