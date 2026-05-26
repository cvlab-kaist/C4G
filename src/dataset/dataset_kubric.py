"""Multi-view Kubric dataset loader (dense temporal sampling).

Data layout (on disk):
    {root}/{rendering_config}/frames/{scene_id:05d}/view_{view_id:04d}/
        rgba_{sub:05d}.png           # RGBA image (512x512, uint8), sub=0..31
        metadata.json                # Camera params (positions[32], quaternions[32], K, ...)

Sampling strategy:
    - Pick N consecutive timestamps (no gap).
    - One randomly chosen timestamp provides all V camera views (context).
    - The remaining N-1 timestamps each provide 1 random view (context).
    - Targets: the V-1 unseen views at each of the N-1 timestamps.

    Context total:  V + (N-1)
    Target total:   (N-1) * (V-1)
"""

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import json
import os

import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


@dataclass
class DatasetKubricCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    rescale_to_1cube: bool
    num_temporal_frames: int = 32
    num_context_timestamps: int = 6
    render_config_include: list[str] | None = None
    val_scenes: list[str] | None = None


@dataclass
class DatasetKubricCfgWrapper:
    kubric: DatasetKubricCfg


class DatasetKubric(Dataset):
    cfg: DatasetKubricCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    near: float = 0.1
    far: float = 100.0

    ALL_VIEWS = ["view_0001", "view_0002", "view_0003", "view_0004"]

    def __init__(self, cfg: DatasetKubricCfg, stage: Stage, view_sampler: ViewSampler):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self.data_root = str(cfg.roots[0])

        # Discover rendering configs (top-level dirs)
        render_configs = sorted([
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
        ])
        if cfg.render_config_include is not None:
            include_set = set(cfg.render_config_include)
            render_configs = [rc for rc in render_configs if rc in include_set]

        # Flatten: each scene = (render_config, scene_id)
        all_scenes: list[tuple[str, str]] = []
        for rc in render_configs:
            frames_dir = os.path.join(self.data_root, rc, "frames")
            if not os.path.isdir(frames_dir):
                continue
            scene_dirs = sorted([
                d for d in os.listdir(frames_dir)
                if os.path.isdir(os.path.join(frames_dir, d))
            ])
            for sd in scene_dirs:
                vdir = os.path.join(frames_dir, sd, "view_0001")
                if os.path.exists(os.path.join(vdir, "rgba_00000.png")):
                    all_scenes.append((rc, sd))

        # Train/val split: last rendering config for val/test, rest for train
        if len(render_configs) >= 2:
            val_configs = set(render_configs[-1:])
            train_configs = set(render_configs[:-1])
        else:
            val_configs = set(render_configs)
            train_configs = set(render_configs)

        if self.stage == "train":
            self.scene_list = [(rc, sd) for rc, sd in all_scenes if rc in train_configs]
        elif self.stage == "val":
            val_all = [(rc, sd) for rc, sd in all_scenes if rc in val_configs]
            step = max(1, len(val_all) // 10)
            self.scene_list = val_all[::step][:10]
        else:
            self.scene_list = [(rc, sd) for rc, sd in all_scenes if rc in val_configs]

        if self.stage == "val" and cfg.val_scenes is not None:
            val_set = set(cfg.val_scenes)
            self.scene_list = [
                (rc, sd) for rc, sd in self.scene_list
                if f"{rc}/{sd}" in val_set or sd in val_set
            ]

        print(f"Kubric [{self.stage}]: {len(self.scene_list)} scenes "
              f"(from {len(render_configs)} render configs, total {len(all_scenes)} available)")

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _view_dir(self, rc: str, scene_id: str, view_name: str) -> str:
        return os.path.join(self.data_root, rc, "frames", scene_id, view_name)

    # ------------------------------------------------------------------
    # Camera conversion: Blender -> OpenCV c2w
    # ------------------------------------------------------------------

    @staticmethod
    def _build_c2w(position, quaternion_wxyz) -> np.ndarray:
        """Convert Blender camera (position + WXYZ quaternion) to OpenCV c2w 4x4."""
        quat_xyzw = [quaternion_wxyz[1], quaternion_wxyz[2],
                      quaternion_wxyz[3], quaternion_wxyz[0]]
        R_bl = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
        flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        R_cv = R_bl @ flip
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_cv
        c2w[:3, 3] = np.array(position, dtype=np.float32)
        return c2w

    @staticmethod
    def _build_intrinsics_pixel(metadata: dict) -> np.ndarray:
        """Build pixel-space 3x3 intrinsics from Blender metadata."""
        focal = metadata["camera"]["focal_length"]
        sensor_w = metadata["camera"]["sensor_width"]
        H, W = 512, 512
        fx = fy = focal / sensor_w * W
        cx, cy = W / 2.0, H / 2.0
        K = np.eye(3, dtype=np.float32)
        K[0, 0], K[1, 1] = fx, fy
        K[0, 2], K[1, 2] = cx, cy
        return K

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    def _load_metadata(self, rc: str, scene_id: str, view_name: str) -> dict:
        path = os.path.join(self._view_dir(rc, scene_id, view_name), "metadata.json")
        with open(path) as f:
            return json.load(f)

    def _load_image(self, rc: str, scene_id: str, view_name: str, sub_idx: int) -> Tensor:
        path = os.path.join(
            self._view_dir(rc, scene_id, view_name), f"rgba_{sub_idx:05d}.png"
        )
        return self.to_tensor(Image.open(path).convert("RGB"))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.scene_list)

    def __getitem__(self, idx: int) -> dict:
        max_retries = 500
        last_exc = None
        for attempt in range(max_retries):
            try:
                return self._getitem_impl(idx)
            except Exception as e:
                last_exc = e
                idx = np.random.randint(len(self))
        raise RuntimeError(
            f"[kubric] Failed after {max_retries} retries. "
            f"Last exception: {type(last_exc).__name__}: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _getitem_impl(self, idx: int) -> dict:
        rc, scene_id = self.scene_list[idx]
        num_views = len(self.ALL_VIEWS)  # V = 4
        num_total = self.cfg.num_temporal_frames  # 32
        N = self.cfg.num_context_timestamps

        # Pick N consecutive timestamps
        max_start = num_total - N
        start = random.randint(0, max_start)
        timestamps = list(range(start, start + N))

        # One random timestamp gets all V views; the rest get 1 random view
        mv_idx = random.randint(0, N - 1)  # index into timestamps
        fixed_ctx_view = random.randint(0, num_views - 1)

        # Pre-load metadata per view
        view_meta: dict[str, tuple] = {}
        for vn in self.ALL_VIEWS:
            meta = self._load_metadata(rc, scene_id, vn)
            K_pixel = self._build_intrinsics_pixel(meta)
            view_meta[vn] = (meta, K_pixel)

        # ---- Build context & target ----
        ctx_cam_ids: list[int] = []
        ctx_ext_list: list[np.ndarray] = []
        ctx_int_list: list[np.ndarray] = []
        ctx_img_specs: list[tuple[str, int]] = []
        ctx_ts_list: list[int] = []

        tgt_cam_ids: list[int] = []
        tgt_ext_list: list[np.ndarray] = []
        tgt_int_list: list[np.ndarray] = []
        tgt_img_specs: list[tuple[str, int]] = []
        tgt_ts_list: list[int] = []

        for i, ts in enumerate(timestamps):
            if i == mv_idx:
                # Multi-view timestamp: all V views go to context
                for c, vn in enumerate(self.ALL_VIEWS):
                    meta, K_pixel = view_meta[vn]
                    cam = meta["camera"]
                    c2w = self._build_c2w(cam["positions"][ts], cam["quaternions"][ts])
                    K_norm = K_pixel.copy()
                    K_norm[0, :] /= 512
                    K_norm[1, :] /= 512
                    ctx_cam_ids.append(c)
                    ctx_ext_list.append(c2w)
                    ctx_int_list.append(K_norm)
                    ctx_img_specs.append((vn, ts))
                    ctx_ts_list.append(ts)
            else:
                # Single-view timestamp: 1 random view to context, V-1 to target
                ctx_view = fixed_ctx_view
                for c, vn in enumerate(self.ALL_VIEWS):
                    meta, K_pixel = view_meta[vn]
                    cam = meta["camera"]
                    c2w = self._build_c2w(cam["positions"][ts], cam["quaternions"][ts])
                    K_norm = K_pixel.copy()
                    K_norm[0, :] /= 512
                    K_norm[1, :] /= 512
                    if c == ctx_view:
                        ctx_cam_ids.append(c)
                        ctx_ext_list.append(c2w)
                        ctx_int_list.append(K_norm)
                        ctx_img_specs.append((vn, ts))
                        ctx_ts_list.append(ts)
                    else:
                        tgt_cam_ids.append(c)
                        tgt_ext_list.append(c2w)
                        tgt_int_list.append(K_norm)
                        tgt_img_specs.append((vn, ts))
                        tgt_ts_list.append(ts)

        # ---- Load images (parallel) ----
        ctx_images = self._load_images_parallel(rc, scene_id, ctx_img_specs)
        tgt_images = self._load_images_parallel(rc, scene_id, tgt_img_specs)

        # ---- Stack camera tensors ----
        ctx_ext = torch.from_numpy(np.stack(ctx_ext_list))
        ctx_int = torch.from_numpy(np.stack(ctx_int_list))
        tgt_ext = torch.from_numpy(np.stack(tgt_ext_list))
        tgt_int = torch.from_numpy(np.stack(tgt_int_list))

        # ---- Normalization ----
        all_ext = torch.cat([ctx_ext, tgt_ext], dim=0)
        n_ctx = len(ctx_cam_ids)

        scale = 1.0
        if self.cfg.make_baseline_1:
            a = all_ext[0, :3, 3]
            diffs = all_ext[1:n_ctx, :3, 3] - a
            norms = diffs.norm(dim=1)
            baseline = norms.max() if norms.numel() > 0 else torch.tensor(0.0)
            if baseline < self.cfg.baseline_min:
                all_diffs = (all_ext[:, :3, 3] - all_ext[0:1, :3, 3]).norm(dim=1)
                if all_diffs.max() < self.cfg.baseline_min:
                    all_ext = torch.eye(4, dtype=all_ext.dtype).unsqueeze(0).expand(all_ext.shape[0], -1, -1).clone()
                else:
                    raise Exception(f"Baseline {baseline:.6f} out of range")
            elif baseline > self.cfg.baseline_max:
                raise Exception(f"Baseline {baseline:.6f} out of range")
            else:
                scale = baseline.item()
                all_ext[:, :3, 3] /= scale

        if self.cfg.relative_pose:
            all_ext = camera_normalization(all_ext[0:1], all_ext)

        if self.cfg.rescale_to_1cube:
            scene_scale = torch.max(torch.abs(all_ext[:n_ctx, :3, 3]))
            if scene_scale > 1e-8:
                all_ext[:, :3, 3] /= scene_scale

        if torch.isnan(all_ext).any() or torch.isinf(all_ext).any():
            raise Exception("NaN or Inf in extrinsics")

        ctx_ext = all_ext[:n_ctx]
        tgt_ext = all_ext[n_ctx:]

        ctx_index = torch.tensor(ctx_ts_list, dtype=torch.int64)
        tgt_index = torch.tensor(tgt_ts_list, dtype=torch.int64)
        ctx_camera = torch.tensor(ctx_cam_ids, dtype=torch.int64)
        tgt_camera = torch.tensor(tgt_cam_ids, dtype=torch.int64)
        n_tgt = len(tgt_cam_ids)

        example = {
            "context": {
                "extrinsics": ctx_ext,
                "intrinsics": ctx_int,
                "image": ctx_images,
                "near": self.get_bound("near", n_ctx) / scale,
                "far": self.get_bound("far", n_ctx) / scale,
                "index": ctx_index,
                "camera": ctx_camera,
            },
            "target": {
                "extrinsics": tgt_ext,
                "intrinsics": tgt_int,
                "image": tgt_images,
                "near": self.get_bound("near", n_tgt) / scale,
                "far": self.get_bound("far", n_tgt) / scale,
                "index": tgt_index,
                "camera": tgt_camera,
            },
            "scene": f"kubric_{rc}_{scene_id}",
            "dataset_name": self.cfg.name,
        }

        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        return apply_crop_shim(example, tuple(self.cfg.input_image_shape))

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_images_parallel(
        self,
        rc: str,
        scene_id: str,
        specs: list[tuple[str, int]],
    ) -> Float[Tensor, "batch 3 height width"]:
        def _load(spec):
            vn, si = spec
            return self._load_image(rc, scene_id, vn, si)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_load, s): j for j, s in enumerate(specs)}
            results = [None] * len(specs)
            for future in futures:
                results[futures[future]] = future.result()
        return torch.stack(results)

    def get_bound(self, bound, num_views):
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage
