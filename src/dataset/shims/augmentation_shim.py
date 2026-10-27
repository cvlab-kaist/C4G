import torch
from jaxtyping import Float
from torch import Tensor

from ..types import AnyExample, AnyViews
from .motion_fields import FLOW_MAP_KEYS, SCALAR_MAP_KEYS


def reflect_extrinsics(
    extrinsics: Float[Tensor, "*batch 4 4"],
) -> Float[Tensor, "*batch 4 4"]:
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    return reflect @ extrinsics @ reflect


def reflect_views(views: AnyViews) -> AnyViews:
    reflected = {
        **views,
        "image": views["image"].flip(-1),
        "extrinsics": reflect_extrinsics(views["extrinsics"]),
    }
    # Per-pixel supervision maps must be mirrored with the images.
    for key in SCALAR_MAP_KEYS:
        if key in views and views[key] is not None:
            reflected[key] = views[key].flip(-1)
    for key in FLOW_MAP_KEYS:
        if key in views and views[key] is not None:
            flow = views[key].flip(-2)  # [..., h, w, 2]: mirror columns
            flow = torch.cat([-flow[..., :1], flow[..., 1:]], dim=-1)  # dx -> -dx
            reflected[key] = flow
    return reflected


def apply_augmentation_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly augment the training images."""
    # Do not augment with 50% chance.
    if torch.rand(tuple(), generator=generator) < 0.5:
        return example

    return {
        **example,
        "context": reflect_views(example["context"]),
        "target": reflect_views(example["target"]),
    }
