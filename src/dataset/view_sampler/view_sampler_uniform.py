from dataclasses import dataclass
from typing import Literal, Optional

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerUniformCfg:
    name: Literal["uniform"]
    num_context_views: int
    warm_up_steps: int
    warm_up_start_steps: int = 0
    num_target_views: int = -1
    initial_gap: int = 3
    gap: int = 3
    val_random_seed: int = 42
    start_index_max: Optional[int] = None

class ViewSamplerUniform(ViewSampler[ViewSamplerUniformCfg]):
    def __init__(
        self,
        cfg: ViewSamplerUniformCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)
        if self.stage == 'val':
            torch.manual_seed(self.cfg.val_random_seed)

    def schedule(self, initial: int, final: int) -> int:
        if self.cfg.warm_up_steps == 0:
            return final
        fraction = (self.global_step - self.cfg.warm_up_start_steps) / (self.cfg.warm_up_steps - self.cfg.warm_up_start_steps)
        fraction = max(0.0, min(1.0, fraction))  # Clamp fraction between 0 and 1
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        assert self.num_context_views >= 2, "At least two context views are required."
        num_views, _, _ = extrinsics.shape
        gap = self.schedule(self.cfg.initial_gap, self.cfg.gap)
        max_start = num_views - gap * self.num_context_views - 1
        if self.cfg.start_index_max is not None:
            max_start = min(max_start, self.cfg.start_index_max - 1)
        start_idx = torch.randint(0, max(max_start, 1), (1,)).item()
        index_context = [start_idx + i * gap for i in range(self.num_context_views)]
        index_target = [start_idx+int(gap/2) + i * gap for i in range(self.num_context_views - 1)]
        index_target = torch.tensor(index_target, dtype=torch.int64, device=device)
        sampled_indices = torch.randperm(index_target.shape[0], device=device)[:self.num_target_views]
        index_target = index_target[sampled_indices]

        overlap = torch.tensor([0.5], device=device)
        return (
            torch.tensor(index_context, dtype=torch.int64, device=device),
            index_target,
            overlap
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_context_views - 1 if self.cfg.num_target_views == -1 else self.cfg.num_target_views