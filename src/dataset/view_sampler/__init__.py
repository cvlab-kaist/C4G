from typing import Any

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from .view_sampler_uniform import ViewSamplerUniform, ViewSamplerUniformCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "bounded": ViewSamplerBounded,
    "uniform": ViewSamplerUniform,
}

ViewSamplerCfg = (
    ViewSamplerBoundedCfg
    | ViewSamplerUniformCfg
)


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )
