"""Model constructor for CELL eval/refinement.

`build_model` instantiates the edge backbone (optionally with feature fusion) and loads a
checkpoint, inferring the confidence-head configuration from the saved weights so shapes match.
Extracted from the original research `track.py` so the eval/refinement entry points depend only
on `core/`.
"""
import torch

from core.backbone import Backbone_Edge_FF, Backbone_Edge


class _Args:
    """Minimal args object supporting attribute access AND `in` (the backbone
    checks `'dropout' not in self.args`)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __contains__(self, k):
        return k in self.__dict__


def build_model(backbone, use_ff, ckpt, device, gpus):
    margs = _Args(mixed_precision=False)
    # infer the conf-head kernel from the checkpoint (if it has one) so shapes match on load
    sd = torch.load(ckpt, map_location="cpu")
    if "module.conf_cfg" in sd:
        # newer checkpoints save the exact head config: [conf_nets, use_net, use_cnet, use_motion, kernel]
        cfg = sd["module.conf_cfg"].tolist()
        margs.conf_nets = int(cfg[0]); margs.conf_kernel = int(cfg[4])
        feats = [n for n, u in zip(["net", "cnet", "motion"], cfg[1:4]) if int(u)]
        margs.conf_feats = ",".join(feats) if feats else "net"
    elif "module.conf_head.conv1.weight" in sd:
        # older net-only checkpoints: infer conf_nets from the input channels
        w = sd["module.conf_head.conv1.weight"]
        margs.conf_kernel = w.shape[-1]
        margs.conf_nets = max(1, w.shape[1] // 128)   # in_channels = 128 * conf_nets
        margs.conf_feats = "net"
    if backbone == "edge" and use_ff:
        net = Backbone_Edge_FF(margs)
    elif backbone == "edge":
        net = Backbone_Edge(margs)
    else:
        raise ValueError("CELL supports --backbone edge (optionally --use_feature_fusion)")
    model = torch.nn.DataParallel(net, device_ids=gpus)
    # strict=False: baseline checkpoints lack conf_head (stays fresh, unused); conf-head
    # checkpoints load it. Any OTHER missing/unexpected key is a real error -> report it.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "conf_head" not in k and "conf_cfg" not in k]
    if real_missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={real_missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model
