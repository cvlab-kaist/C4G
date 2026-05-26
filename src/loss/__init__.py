from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_dynamicmask import LossDynamicMask, LossDynamicMaskCfgWrapper

LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossDynamicMaskCfgWrapper: LossDynamicMask,  # Wrapper를 키로 사용
}

LossCfgWrapper = LossLpipsCfgWrapper | LossMseCfgWrapper | LossDynamicMaskCfgWrapper

def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]