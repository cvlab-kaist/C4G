from typing import Any
import torch.nn as nn

from .backbone import Backbone
from .backbone_croco import AsymmetricCroCo, BackboneCrocoCfg
from .backbone_vggt import BackboneVGGT

BACKBONES: dict[str, Backbone[Any]] = {
    "croco": AsymmetricCroCo,
    "vggt_multi": BackboneVGGT,
}

BackboneCfg = BackboneCrocoCfg


def get_backbone(cfg: BackboneCfg, d_in: int = 3, gradient_checkpoint: bool = False) -> nn.Module:
    if gradient_checkpoint:
        return BACKBONES[cfg.name](cfg, d_in, gradient_checkpoint=gradient_checkpoint)
    else:
        return BACKBONES[cfg.name](cfg, d_in)
