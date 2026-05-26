from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossDynamicMaskCfg:
    weight_static: float
    weight_dynamic: float

@dataclass
class LossDynamicMaskCfgWrapper:
    dynamicmask: LossDynamicMaskCfg


class LossDynamicMask(Loss[LossDynamicMaskCfg, LossDynamicMaskCfgWrapper]):
    has_dynamic_mask = True
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image=None,
        dynamic_mask=None,
    ) -> Float[Tensor, ""]:
        delta = prediction.color - target_image
        mse = delta**2
        
        if dynamic_mask is not None:
            if dynamic_mask.dim() == 3:  # [V, H, W]
                dynamic_mask = dynamic_mask.unsqueeze(0).unsqueeze(2)  # [1, V, 1, H, W]
            elif dynamic_mask.dim() == 4:  # [B, V, H, W]
                dynamic_mask = dynamic_mask.unsqueeze(2)  # [B, V, 1, H, W]

            dynamic_mask = dynamic_mask.to(device=mse.device, dtype=mse.dtype)
            if dynamic_mask.shape[-2:] != mse.shape[-2:]:
                leading_shape = dynamic_mask.shape[:-2]
                dynamic_mask = dynamic_mask.reshape(-1, 1, *dynamic_mask.shape[-2:])
                dynamic_mask = F.interpolate(
                    dynamic_mask,
                    size=mse.shape[-2:],
                    mode="nearest",
                )
                dynamic_mask = dynamic_mask.reshape(*leading_shape, *mse.shape[-2:])
            
            valid_mask = (dynamic_mask >= 0).float()
            # dynamic_mask = dynamic_mask.clamp(min=0)
            
            static_mask = (1 - dynamic_mask) * valid_mask
            dynamic_mask = dynamic_mask * valid_mask
            mse_static = mse * static_mask
            mse_dynamic = mse * dynamic_mask
            
            loss = self.cfg.weight_static * mse_static.sum() / (static_mask.sum() + 1e-8) + \
                   self.cfg.weight_dynamic * mse_dynamic.sum() / (dynamic_mask.sum() + 1e-8)
            return loss
        
        return (self.cfg.weight_static + self.cfg.weight_dynamic) * mse.mean()
