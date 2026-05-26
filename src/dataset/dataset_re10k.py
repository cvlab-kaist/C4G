import json
import os
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    fake_time_labels: bool = True


@dataclass
class DatasetRE10kCfgWrapper:
    re10k: DatasetRE10kCfg


@dataclass
class DatasetDL3DVCfgWrapper:
    dl3dv: DatasetRE10kCfg


@dataclass
class DatasetScannetppCfgWrapper:
    scannetpp: DatasetRE10kCfg


class DatasetRE10k(Dataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

        index = self.index
        rank = int(os.environ.get("LOCAL_RANK", 0))

        if self.cfg.overfit_to_scene is not None:
            self._flat_index = [(index[self.cfg.overfit_to_scene], self.cfg.overfit_to_scene)] * len(self.chunks)
        else:
            chunk_paths = set(self.chunks)
            self._flat_index = [
                (chunk_path, scene_key)
                for scene_key, chunk_path in index.items()
                if chunk_path in chunk_paths
            ]

        print(f"re10k: {self.stage}: indexed {len(self._flat_index)} scenes "
              f"from {len(self.chunks)} chunks (rank {rank})")

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __len__(self) -> int:
        return len(self._flat_index)

    def __getitem__(self, idx: int) -> dict:
        max_retries = 50
        for attempt in range(max_retries):
            try:
                return self._getitem(idx)
            except Exception:
                idx = np.random.randint(len(self))
        raise RuntimeError(f"Failed to load a valid sample after {max_retries} retries")

    def _getitem(self, idx: int) -> dict:
        chunk_path, scene_key = self._flat_index[idx]
        chunk = torch.load(chunk_path, weights_only=False)

        item = [x for x in chunk if x["key"] == scene_key]
        assert len(item) == 1
        example = item[0]

        del chunk

        extrinsics, intrinsics = self.convert_poses(example["cameras"])
        scene = example["key"]

        context_indices, target_indices, overlap = self.view_sampler.sample(
            scene,
            extrinsics,
            intrinsics,
        )

        # Skip the example if the field of view is too wide.
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise Exception("FOV too wide")

        # Load the images.
        context_images = [
            example["images"][index.item()] for index in context_indices
        ]
        context_images = self.convert_images(context_images)
        target_images = [
            example["images"][index.item()] for index in target_indices
        ]
        target_images = self.convert_images(target_images)

        # Skip the example if the images don't have the right shape.
        context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
        target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
        if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
            raise Exception("Bad image shape")

        # Resize the world to make the baseline 1.
        context_extrinsics = extrinsics[context_indices]
        if self.cfg.make_baseline_1:
            a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                raise Exception(f"Baseline out of range: {scale:.6f}")
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        if self.cfg.relative_pose:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

        num_ctx = len(context_indices)
        num_tgt = len(target_indices)

        if self.cfg.fake_time_labels:
            # Create diverse fake time labels for temporal embedding.
            max_time = torch.randint(
                num_ctx + num_tgt + 2,
                max(num_ctx + num_tgt + 3, 100),
                (1,),
            ).item()
            fake_context_time = torch.randperm(max_time)[:num_ctx]
            ctx_min = fake_context_time.min().item()
            ctx_max = fake_context_time.max().item()
            if ctx_max - ctx_min < 2:
                fake_context_time[fake_context_time.argmax()] = ctx_min + num_tgt + 1
                ctx_max = ctx_min + num_tgt + 1
            fake_target_time = torch.randint(ctx_min + 1, ctx_max, (num_tgt,))

            # Sort context by fake time so encoding gets temporally coherent views.
            sort_order = fake_context_time.argsort()
            fake_context_time = fake_context_time[sort_order]
            context_indices = context_indices[sort_order]
            context_images = context_images[sort_order]
        else:
            # Use actual frame indices as time labels (unmixed).
            sort_order = context_indices.argsort()
            context_indices = context_indices[sort_order]
            context_images = context_images[sort_order]
            fake_context_time = context_indices.clone()
            fake_target_time = target_indices.clone()

        result = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", num_ctx) / scale,
                "far": self.get_bound("far", num_ctx) / scale,
                "index": fake_context_time,
                "camera": torch.zeros(num_ctx, dtype=torch.long),
                "overlap": overlap,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", num_tgt) / scale,
                "far": self.get_bound("far", num_tgt) / scale,
                "index": fake_target_time,
                "camera": torch.zeros(num_tgt, dtype=torch.long),
            },
            "scene": scene,
            "dataset_name": self.cfg.name,
        }
        if self.stage == "train" and self.cfg.augment:
            result = apply_augmentation_shim(result)
        return apply_crop_shim(result, tuple(self.cfg.input_image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index
