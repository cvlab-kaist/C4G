from typing import Any
import torch
import torch.nn as nn

from .backbone import Backbone
from .backbone_vggt import BackboneVGGT, BackboneVGGTCfg

# TF32 matmuls (gpu >= Ampere). This global flag used to be set as an import side effect of the
# removed CroCo backbone module; it is kept here so training and inference numerics are unchanged.
torch.backends.cuda.matmul.allow_tf32 = True

BACKBONES: dict[str, Backbone[Any]] = {
    "vggt_multi": BackboneVGGT,
}

BackboneCfg = BackboneVGGTCfg


def get_backbone(cfg: BackboneCfg, d_in: int = 3, gradient_checkpoint: bool = False) -> nn.Module:
    if gradient_checkpoint:
        return BACKBONES[cfg.name](cfg, d_in, gradient_checkpoint=gradient_checkpoint)
    else:
        return BACKBONES[cfg.name](cfg, d_in)
