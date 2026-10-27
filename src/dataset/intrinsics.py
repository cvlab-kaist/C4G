import torch
from jaxtyping import Float
from torch import Tensor

from .types import AnyExample, AnyViews


DAVIS_RE10K_INTRINSIC = torch.tensor(
    [
        [0.8767, 0.0, 0.5],
        [0.0, 0.8767, 0.5],
        [0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


def davis_intrinsics_like(
    intrinsics: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch 3 3"]:
    intrinsic = DAVIS_RE10K_INTRINSIC.to(
        device=intrinsics.device,
        dtype=intrinsics.dtype,
    )
    return intrinsic.expand(*intrinsics.shape[:-2], -1, -1).clone()


def apply_davis_intrinsics_to_views(views: AnyViews) -> AnyViews:
    return {
        **views,
        "intrinsics": davis_intrinsics_like(views["intrinsics"]),
    }


def apply_davis_intrinsics(example: AnyExample) -> AnyExample:
    return {
        **example,
        "context": apply_davis_intrinsics_to_views(example["context"]),
        "target": apply_davis_intrinsics_to_views(example["target"]),
    }


def maybe_apply_davis_intrinsics(
    example: AnyExample,
    enabled: bool,
) -> AnyExample:
    return apply_davis_intrinsics(example) if enabled else example
