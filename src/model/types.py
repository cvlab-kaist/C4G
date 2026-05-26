from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian dim"] | None = None
    rotations: Float[Tensor, "batch gaussian 4"] | None = None


def merge_gaussians(a: Gaussians, b: Gaussians) -> Gaussians:
    """Merge two Gaussians by concatenating along the gaussian dimension (dim=1)."""
    return Gaussians(
        means=torch.cat([a.means, b.means], dim=1),
        covariances=torch.cat([a.covariances, b.covariances], dim=1),
        harmonics=torch.cat([a.harmonics, b.harmonics], dim=1),
        opacities=torch.cat([a.opacities, b.opacities], dim=1),
        scales=torch.cat([a.scales, b.scales], dim=1) if a.scales is not None and b.scales is not None else None,
        rotations=torch.cat([a.rotations, b.rotations], dim=1) if a.rotations is not None and b.rotations is not None else None,
    )
