"""EgoExo4D dataset loader (mono-video, same-camera targets — like Spring).

Sampling strategy:
    - Randomly pick one exo camera per take.
    - view_sampler picks context & target *timestamps*.
    - Context and target are all from the same single camera.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from unittest import mock

import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .intrinsics import davis_intrinsics_like, maybe_apply_davis_intrinsics
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None


@dataclass
class DatasetEgoExo4DMonoCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    frame_stride: int = 10

    # Share of the mixed training sampler given to this dataset.
    sample_weight: float = 1.0
    # How takes are drawn inside the dataset: uniform, or by the (sqrt of the)
    # number of view-sampler windows of the take.
    item_weighting: str = "uniform"  # uniform | sqrt | linear


@dataclass
class DatasetEgoExo4DMonoCfgWrapper:
    egoexo4d_mono: DatasetEgoExo4DMonoCfg


class DatasetEgoExo4DMono(Dataset):
    """EgoExo4D mono-video loader: single random exo camera, same camera for targets."""

    cfg: DatasetEgoExo4DMonoCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetEgoExo4DMonoCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self.takes: list[dict] = []

        for root in self.cfg.roots:
            root = Path(root)
            self._load_root(root)

        print(f"egoexo4d_mono: {self.stage}: {len(self.takes)} takes")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stage_to_split(stage: Stage) -> str:
        if stage in ("train", "val", "test"):
            return stage
        return "train"

    def _load_root(self, root: Path) -> None:
        split = self._stage_to_split(self.stage)

        with open(root / "annotations" / "splits.json") as f:
            uid_to_split: dict[str, str] = json.load(f)["take_uid_to_split"]

        with open(root / "takes.json") as f:
            takes_list: list[dict] = json.load(f)
        uid_to_take = {t["take_uid"]: t for t in takes_list}

        cam_pose_dir = root / "annotations" / "ego_pose" / split / "camera_pose"

        for take_uid, take_split in uid_to_split.items():
            if take_split != split:
                continue
            if take_uid not in uid_to_take:
                continue

            take_meta = uid_to_take[take_uid]
            take_name = take_meta["take_name"]

            cam_pose_path = cam_pose_dir / f"{take_uid}.json"
            if not cam_pose_path.exists():
                continue

            video_dir = root / "takes" / take_name / "frame_aligned_videos" / "downscaled" / "448"
            if not video_dir.exists():
                continue

            # Avoid loading every multi-MB camera JSON during dataloader startup.
            # Full camera metadata is parsed lazily in __getitem__.
            is_dance = "dance" in take_name.lower()
            has_exo = any(
                p.stem.startswith("cam") and (not is_dance or p.stem < "cam05")
                for p in video_dir.glob("*.mp4")
            )
            if not has_exo:
                continue

            # Estimate num_frames without parsing the full camera JSON
            dur = take_meta.get("duration_sec", 0)
            num_frames = int(dur * 30) if dur > 0 else 0
            if num_frames < 2:
                continue

            self.takes.append({
                "take_uid": take_uid,
                "take_name": take_name,
                "video_dir": video_dir,
                "cam_pose_path": cam_pose_path,
                "num_frames": num_frames,
            })

    @staticmethod
    def _parse_cameras(
        cam_data: dict,
        video_dir: Path,
        take_name: str = "",
    ) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
        # Skip cam05+ for dance takes (top-down overhead cameras)
        is_dance = "dance" in take_name.lower()

        exo_cams: list[str] = []
        extrinsics: dict[str, np.ndarray] = {}
        intrinsics: dict[str, np.ndarray] = {}

        for key in sorted(cam_data.keys()):
            if key in ("metadata",) or key.startswith("aria"):
                continue
            if is_dance and key >= "cam05":
                continue
            if not (video_dir / f"{key}.mp4").exists():
                continue

            entry = cam_data[key]
            ext_34 = np.array(entry["camera_extrinsics"], dtype=np.float32)
            K = np.array(entry["camera_intrinsics"], dtype=np.float32)

            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :] = ext_34
            c2w = np.linalg.inv(w2c).astype(np.float32)

            native_w = K[0, 2] * 2
            native_h = K[1, 2] * 2
            K[0, :] /= native_w
            K[1, :] /= native_h

            exo_cams.append(key)
            extrinsics[key] = c2w
            intrinsics[key] = K

        return exo_cams, extrinsics, intrinsics

    @staticmethod
    def _get_frame_count_from_pose(cam_data: dict) -> int | None:
        for key in cam_data:
            if not key.startswith("aria"):
                continue
            ext = cam_data[key].get("camera_extrinsics", {})
            if isinstance(ext, dict) and len(ext) > 0:
                return len(ext)
        return None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.takes)

    def item_weights(self) -> list[float]:
        sampler_cfg = getattr(self.view_sampler, "cfg", None)
        gap = getattr(sampler_cfg, "gap", 1)
        span = gap * self.view_sampler.num_context_views + 1
        weights = []
        for take in self.takes:
            windows = take["num_frames"] // self.cfg.frame_stride - span
            windows = float(max(windows, 1))
            if self.cfg.item_weighting == "sqrt":
                weights.append(windows ** 0.5)
            elif self.cfg.item_weighting == "linear":
                weights.append(windows)
            else:
                weights.append(1.0)
        return weights

    def __getitem__(self, idx: int) -> dict:
        max_retries = 50
        last_exc = None
        for attempt in range(max_retries):
            try:
                return self._getitem_impl(idx)
            except Exception as e:
                last_exc = e
                idx = np.random.randint(len(self))
        raise RuntimeError(
            f"Failed to load a valid sample after {max_retries} retries. "
            f"Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _getitem_impl(self, idx: int) -> dict:
        take = self.takes[idx]
        video_dir: Path = take["video_dir"]
        total_frames: int = take["num_frames"]

        # Lazy-parse camera data
        with open(take["cam_pose_path"]) as f:
            cam_data: dict = json.load(f)
        exo_cams, extrinsics_dict, intrinsics_dict = self._parse_cameras(
            cam_data, video_dir, take["take_name"]
        )
        if len(exo_cams) < 1:
            raise Exception("No valid exo cameras")

        # Update num_frames from pose data if available
        num_frames_from_pose = self._get_frame_count_from_pose(cam_data)
        if num_frames_from_pose is not None:
            total_frames = num_frames_from_pose

        # Randomly pick one exo camera (mono-video)
        cam_name = random.choice(exo_cams)

        stride = self.cfg.frame_stride
        num_timestamps = total_frames // stride
        # Uniform sampler needs: gap * num_context_views + 1 timestamps
        min_timestamps = self.view_sampler.cfg.gap * self.view_sampler.num_context_views + 1
        if num_timestamps < min_timestamps:
            raise Exception(
                f"Not enough timestamps ({num_timestamps}, need {min_timestamps}) for "
                f"{self.view_sampler.num_context_views} context views"
            )

        # Static camera: same extrinsics/intrinsics for all timestamps
        c2w = extrinsics_dict[cam_name]  # (4,4)
        K = intrinsics_dict[cam_name]  # (3,3)

        # view_sampler expects (num_views, 4, 4) extrinsics
        ext_all = torch.tensor(c2w, dtype=torch.float32).unsqueeze(0).expand(num_timestamps, -1, -1)
        int_all = torch.tensor(K, dtype=torch.float32).unsqueeze(0).expand(num_timestamps, -1, -1)

        ctx_ts, tgt_ts, overlap = self.view_sampler.sample("", ext_all, int_all)

        ctx_frame_indices = (ctx_ts * stride).tolist()
        tgt_frame_indices = (tgt_ts * stride).tolist()

        # Load images from the single camera
        video_path = str(video_dir / f"{cam_name}.mp4")
        ctx_images = self._load_video_frames(video_path, ctx_frame_indices)
        tgt_images = self._load_video_frames(video_path, tgt_frame_indices)

        if self.cfg.skip_bad_shape:
            expected = (3, *self.cfg.original_image_shape)
            if ctx_images.shape[1:] != expected or tgt_images.shape[1:] != expected:
                raise Exception("Bad image shape")

        # Static camera: use identity extrinsics for all frames
        n_ctx = len(ctx_ts)
        n_tgt = len(tgt_ts)
        ctx_ext = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(n_ctx, -1, -1).clone()
        tgt_ext = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(n_tgt, -1, -1).clone()

        # Override intrinsics with RE10k values (model is overfitted to re10k)
        re10k_int = torch.tensor(
            [[0.4836, 0.0, 0.5], [0.0, 0.8597, 0.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        ctx_int = re10k_int.unsqueeze(0).expand(n_ctx, -1, -1).clone()
        tgt_int = re10k_int.unsqueeze(0).expand(n_tgt, -1, -1).clone()
        if self.cfg.force_davis_intrinsics:
            ctx_int = davis_intrinsics_like(ctx_int)
            tgt_int = davis_intrinsics_like(tgt_int)

        scale = 1.0

        scene = f"egoexo4d_mono/{take['take_name']}/{cam_name}"

        example = {
            "context": {
                "extrinsics": ctx_ext,
                "intrinsics": ctx_int,
                "image": ctx_images,
                "near": self.get_bound("near", n_ctx) / scale,
                "far": self.get_bound("far", n_ctx) / scale,
                "index": ctx_ts,
                "camera": torch.zeros_like(ctx_ts),  # single camera
            },
            "target": {
                "extrinsics": tgt_ext,
                "intrinsics": tgt_int,
                "image": tgt_images,
                "near": self.get_bound("near", n_tgt) / scale,
                "far": self.get_bound("far", n_tgt) / scale,
                "index": tgt_ts,
                "camera": torch.zeros_like(tgt_ts),  # single camera
            },
            "scene": scene,
            "dataset_name": self.cfg.name,
        }

        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        example = apply_crop_shim(example, tuple(self.cfg.input_image_shape))
        return maybe_apply_davis_intrinsics(example, self.cfg.force_davis_intrinsics)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_video_frames(
        self,
        video_path: str,
        frame_indices: list[int],
    ) -> Float[Tensor, "batch 3 height width"]:
        images = [self._read_frame(video_path, fi) for fi in frame_indices]
        return self._convert_numpy_images(images)

    def _read_frame(self, video_path: str, index: int) -> np.ndarray:
        if iio is None:
            raise ImportError("imageio.v3 is required to read video frames.")
        mock.patch("imageio_ffmpeg._io.subprocess.Popen.kill").start()
        frame = iio.imread(video_path, index=index)  # H W 3 uint8
        H, W = self.cfg.original_image_shape
        img = Image.fromarray(frame)
        if img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)
        return np.asarray(img)

    def _convert_numpy_images(
        self,
        images: list[np.ndarray],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for im in images:
            if im.ndim == 3 and im.shape[2] == 4:
                im = im[:, :, :3]
            torch_images.append(self.to_tensor(Image.fromarray(im)))
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
            return "train"
        return "train" if self.stage in ("train", "val") else self.stage
