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
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .intrinsics import davis_intrinsics_like, maybe_apply_davis_intrinsics
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

    # Share of the mixed training sampler given to this dataset.
    sample_weight: float = 1.0
    # Ship GT depth / 2D flow / next-frame depth / dynamic mask for every view
    # (see shims/motion_fields.py); consumed by the motion losses.
    motion_fields: bool = False
    # An instance counts as moving at frame t when one of its 3D bbox corners
    # travels more than this (scene units) to the previous or next frame.
    motion_instance_threshold: float = 0.02
    # Also expose the motion mask as the photometric `mask` (LossDynamicMask).
    photometric_mask: bool = False
    # Ablation: no all-view timestamp. Context = the one fixed camera at every timestamp (a monocular clip);
    # targets = the other V-1 cameras at every timestamp (context N, targets N*(V-1)).
    monocular_context: bool = False


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
    # Motion supervision maps
    # ------------------------------------------------------------------

    def _load_flow(self, rc: str, scene_id: str, view_name: str, sub_idx: int, direction: str = "forward") -> Tensor:
        """GT optical flow as [h, w, 2] (dx, dy) in 512-res pixels.

        Kubric writes the flow as an 8-bit RGB png with the (dy, dx) range in
        data_ranges.json (channel 0 is the row displacement).
        """
        vdir = self._view_dir(rc, scene_id, view_name)
        png = np.asarray(Image.open(os.path.join(vdir, f"{direction}_flow_{sub_idx:05d}.png")), dtype=np.float32)
        with open(os.path.join(vdir, "data_ranges.json")) as f:
            rng = json.load(f)[f"{direction}_flow"]
        flow = png[..., :2] / 255.0 * (rng["max"] - rng["min"]) + rng["min"]
        return torch.from_numpy(np.ascontiguousarray(flow[..., ::-1]))  # (dy, dx) -> (dx, dy)

    def _load_depth(self, rc: str, scene_id: str, view_name: str, sub_idx: int) -> Tensor:
        path = os.path.join(self._view_dir(rc, scene_id, view_name), f"depth_{sub_idx:05d}.tiff")
        d = torch.tensor(np.array(Image.open(path)), dtype=torch.float32)
        return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    def _load_segmentation(self, rc: str, scene_id: str, view_name: str, sub_idx: int) -> Tensor:
        path = os.path.join(self._view_dir(rc, scene_id, view_name), f"segmentation_{sub_idx:05d}.png")
        return torch.from_numpy(np.array(Image.open(path)).astype(np.int64))

    @staticmethod
    def _instance_motion(instances: list[dict], threshold: float) -> np.ndarray:
        """[num_instances + 1, num_frames] bool: is segmentation id moving at frame t.

        Id 0 is the (static) background. An instance moves at t when a 3D bbox
        corner travels more than `threshold` between t-1 -> t or t -> t+1, so
        collision-nudged "static" objects are excluded from the static set too.
        """
        num_frames = len(instances[0]["bboxes_3d"]) if instances else 0
        moving = np.zeros((len(instances) + 1, num_frames), dtype=bool)
        for k, inst in enumerate(instances):
            corners = np.asarray(inst["bboxes_3d"], dtype=np.float64)  # [T, 8, 3]
            step = np.abs(corners[1:] - corners[:-1]).max(axis=(1, 2)) > threshold  # [T-1]
            moving[k + 1, :-1] |= step
            moving[k + 1, 1:] |= step
        return moving

    @staticmethod
    def _sample_map(img: Tensor, coords: Tensor, edge_tol: float = 0.05) -> tuple[Tensor, Tensor]:
        """Sample depth img [h, w] at continuous pixel coords [h, w, 2].

        Nearest-neighbour value; a sample is invalid outside the image or where
        the four surrounding pixels disagree by more than `edge_tol` (relative),
        i.e. on a depth discontinuity, where any resampling would mix surfaces.
        """
        h, w = img.shape
        grid = torch.stack(
            [2.0 * coords[..., 0] / w - 1.0, 2.0 * coords[..., 1] / h - 1.0], dim=-1
        )[None]
        nearest = F.grid_sample(img[None, None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0]
        x = coords[..., 0] - 0.5
        y = coords[..., 1] - 0.5
        x0 = x.floor().long().clamp(0, w - 1)
        y0 = y.floor().long().clamp(0, h - 1)
        x1 = (x0 + 1).clamp(max=w - 1)
        y1 = (y0 + 1).clamp(max=h - 1)
        corners = torch.stack([img[y0, x0], img[y0, x1], img[y1, x0], img[y1, x1]])
        spread = (corners.max(0).values - corners.min(0).values) / nearest.clamp(min=1e-6)
        inside = (grid[0].abs() <= 1.0).all(dim=-1) & (spread <= edge_tol)
        return nearest, inside

    def _load_motion_fields(
        self,
        rc: str,
        scene_id: str,
        specs: list[tuple[str, int]],
        batch_timestamps: set[int],
        instances: list[dict],
        depth_scale: float,
    ) -> dict[str, Tensor]:
        """Per-view depth / flow / next depth / dynamic mask at 512x512."""
        moving = self._instance_motion(instances, self.cfg.motion_instance_threshold)
        num_total = self.cfg.num_temporal_frames
        H = W = 512
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij"
        )
        pixel_centers = torch.stack([xx + 0.5, yy + 0.5], dim=-1)

        def _one(spec):
            vn, ts = spec
            depth = self._load_depth(rc, scene_id, vn, ts)
            seg = self._load_segmentation(rc, scene_id, vn, ts)
            mask = torch.from_numpy(moving[:, ts])[seg.clamp(max=moving.shape[0] - 1)].float()
            has_next = (ts + 1) in batch_timestamps and ts + 1 < num_total
            if has_next:
                flow = self._load_flow(rc, scene_id, vn, ts, "forward")
                depth_next_full = self._load_depth(rc, scene_id, vn, ts + 1)
                depth_next, inside = self._sample_map(depth_next_full, pixel_centers + flow)
                valid = (inside & (depth > 0) & (depth_next > 0)).float()
            else:
                flow = torch.zeros(H, W, 2)
                depth_next = torch.zeros(H, W)
                valid = torch.zeros(H, W)
            return depth / depth_scale, flow, depth_next / depth_scale, mask, valid

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(_one, specs))
        depth, flow, depth_next, mask, valid = (torch.stack(x) for x in zip(*results))
        return {
            "depth": depth,
            "motion_flow": flow,
            "motion_depth_next": depth_next,
            "motion_mask": mask,
            "motion_valid": valid,
        }

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
        if self.cfg.monocular_context:
            mv_idx = -1  # no all-view timestamp
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
        if self.cfg.force_davis_intrinsics:
            ctx_int = davis_intrinsics_like(ctx_int)
            tgt_int = davis_intrinsics_like(tgt_int)

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

        depth_scale = float(scale)
        if self.cfg.rescale_to_1cube:
            scene_scale = torch.max(torch.abs(all_ext[:n_ctx, :3, 3]))
            if scene_scale > 1e-8:
                all_ext[:, :3, 3] /= scene_scale
                depth_scale *= float(scene_scale)

        if torch.isnan(all_ext).any() or torch.isinf(all_ext).any():
            raise Exception("NaN or Inf in extrinsics")

        ctx_ext = all_ext[:n_ctx]
        tgt_ext = all_ext[n_ctx:]

        ctx_motion, tgt_motion = {}, {}
        if self.cfg.motion_fields:
            instances = view_meta[self.ALL_VIEWS[0]][0]["instances"]
            batch_ts = set(timestamps)
            ctx_motion = self._load_motion_fields(rc, scene_id, ctx_img_specs, batch_ts, instances, depth_scale)
            tgt_motion = self._load_motion_fields(rc, scene_id, tgt_img_specs, batch_ts, instances, depth_scale)
            if self.cfg.photometric_mask:
                ctx_motion["mask"] = ctx_motion["motion_mask"]
                tgt_motion["mask"] = tgt_motion["motion_mask"]

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
                **ctx_motion,
            },
            "target": {
                "extrinsics": tgt_ext,
                "intrinsics": tgt_int,
                "image": tgt_images,
                "near": self.get_bound("near", n_tgt) / scale,
                "far": self.get_bound("far", n_tgt) / scale,
                "index": tgt_index,
                "camera": tgt_camera,
                **tgt_motion,
            },
            "scene": f"kubric_{rc}_{scene_id}",
            "dataset_name": self.cfg.name,
        }

        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        example = apply_crop_shim(example, tuple(self.cfg.input_image_shape))
        return maybe_apply_davis_intrinsics(example, self.cfg.force_davis_intrinsics)

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
