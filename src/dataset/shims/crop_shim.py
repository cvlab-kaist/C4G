import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from ..types import AnyExample, AnyViews
from .motion_fields import FLOW_MAP_KEYS, SCALAR_MAP_KEYS


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)


def rescale_and_crop_depth(
    depth: Float[Tensor, "*#batch h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "*#batch h_out w_out"]:
    """Apply the same rescale + center crop as rescale_and_crop to a depth map.

    Nearest-neighbour resampling keeps depth values and the zero (= invalid)
    markers intact instead of blending them.
    """
    *batch, h_in, w_in = depth.shape
    h_out, w_out = shape
    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    x = depth.reshape(-1, 1, h_in, w_in)
    x = F.interpolate(x, size=(h_scaled, w_scaled), mode="nearest")
    row = (h_scaled - h_out) // 2
    col = (w_scaled - w_out) // 2
    x = x[:, 0, row : row + h_out, col : col + w_out]
    return x.reshape(*batch, h_out, w_out)


def rescale_and_crop_flow(
    flow: Float[Tensor, "*#batch h_in w_in 2"],
    shape: tuple[int, int],
) -> Float[Tensor, "*#batch h_out w_out 2"]:
    """Apply the image rescale + center crop to a (dx, dy) pixel flow map.

    The flow values are scaled by the same factors as the pixel grid; a center
    crop does not change displacements.
    """
    *batch, h_in, w_in, _ = flow.shape
    h_out, w_out = shape
    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    x = flow.reshape(-1, h_in, w_in, 2).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(h_scaled, w_scaled), mode="nearest")
    row = (h_scaled - h_out) // 2
    col = (w_scaled - w_out) // 2
    x = x[:, :, row : row + h_out, col : col + w_out].permute(0, 2, 3, 1)
    x = x * torch.tensor(
        [w_scaled / w_in, h_scaled / h_in], dtype=x.dtype, device=x.device
    )
    return x.reshape(*batch, h_out, w_out, 2)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape)
    cropped = {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }
    # Per-pixel supervision maps follow the images (each from its own
    # resolution, so precomputed maps may already be stored pre-cropped).
    for key in SCALAR_MAP_KEYS:
        if key in views and views[key] is not None:
            cropped[key] = rescale_and_crop_depth(views[key], shape)
    for key in FLOW_MAP_KEYS:
        if key in views and views[key] is not None:
            cropped[key] = rescale_and_crop_flow(views[key], shape)
    return cropped


def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
    }
