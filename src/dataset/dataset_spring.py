import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization
from ..misc.dynamics import flow_to_dynamic_mask


@dataclass
class DatasetSpringCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    mask_root: Path
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool


@dataclass
class DatasetSpringCfgWrapper:
    spring: DatasetSpringCfg

class DatasetSpring(Dataset):
    cfg: DatasetSpringCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0
    
    def __init__(
        self,
        cfg: DatasetSpringCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        
        # load data
        self.data_root = os.path.join(cfg.roots[0], "train")
        self.data_list = []
        
        if self.data_stage == "train":
            self.data_list = os.listdir(self.data_root)
            self.data_list.remove("0001")
            self.data_list.remove("0002")
            self.data_list.remove("0004")
            self.data_list.remove("0005")
        else:
            # self.data_list = ["0001", "0002", "0004", "0005"]
            self.data_list = os.listdir(self.data_root)
        

        self.scene_ids = {}
        self.scenes = {}
        index = 0
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(self.load_data, os.path.join(self.data_root, scene_path)) for scene_path in self.data_list]
            for future in as_completed(futures):
                scene_frames, scene_id = future.result()
                self.scenes[scene_id] = scene_frames
                self.scene_ids[index] = scene_id
                index += 1
        print(f"Spring: {self.stage}: loaded {len(self.scene_ids)} scenes")

    def readFlo5Flow(self, filename):
        with h5py.File(filename, "r") as f:
            if "flow" not in f.keys():
                raise IOError(f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?")
            return f["flow"][()]

    def readDsp5Disp(self, filename):
        with h5py.File(filename, "r") as f:
            if "disparity" not in f.keys():
                raise IOError(f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?")
            return f["disparity"][()]


    def load_data(self, scene_path):
        scene_id = os.path.basename(scene_path)

        data = []
        extr_path = os.path.join(scene_path, "cam_data", "extrinsics.txt")
        intr_path = os.path.join(scene_path, "cam_data", "intrinsics.txt")
        extrinsics = np.loadtxt(extr_path).reshape(-1, 4, 4)
        intrinsics_original = np.loadtxt(intr_path).reshape(-1, 4)
        intrinsics = np.eye(3, dtype=np.float32)[None, :, :].repeat(extrinsics.shape[0], axis=0)
        intrinsics[:, 0, 0] = intrinsics_original[:, 0]
        intrinsics[:, 1, 1] = intrinsics_original[:, 1]
        intrinsics[:, 0, 2] = intrinsics_original[:, 2]
        intrinsics[:, 1, 2] = intrinsics_original[:, 3]
        intrinsics[:, 0, :] /= self.cfg.original_image_shape[1]
        intrinsics[:, 1, :] /= self.cfg.original_image_shape[0]

        for i in range(extrinsics.shape[0]):
            frame = {
                "rgb_file_path": os.path.join(scene_path, "frame_left", f"frame_left_{i+1:04d}.png"),
                "disp_file_path": os.path.join(scene_path, "disp1_left", f"disp1_left_{i+1:04d}.dsp5"),
                "flow_file_path": os.path.join(scene_path, "flow_FW_left", f"flow_FW_left_{i+1:04d}.flo5")
                                    if i != extrinsics.shape[0] - 1 else None,
                "extrinsics": extrinsics[i],
                "intrinsics": intrinsics[i],
            }
            data.append(frame)
        return data, scene_id

    def load_depths(self, frames, intrinsic):
        depths = [None] * len(frames)
        for idx, frame in enumerate(frames):
            disp_file_path = frame["disp_file_path"]
            if disp_file_path is None:
                continue
            disp = self.to_tensor(self.readDsp5Disp(disp_file_path)).to(torch.float32)
            # Convert disparity to depth
            depth = torch.where(disp > 0, 0.065 * intrinsic[0, 0] * disp.shape[1] / (disp), torch.tensor(1e5))  # Avoid division by zero
            depths[idx] = depth
        return torch.stack(depths).squeeze(1)

    def load_flows(self, frames):
        flows = [None] * (len(frames) - 1)
        for idx, frame in enumerate(frames):
            flow_file_path = frame["flow_file_path"]
            if flow_file_path is None:
                continue
            flow = self.to_tensor(self.readFlo5Flow(flow_file_path)).to(torch.float32)
            flows[idx] = flow
        flows = torch.stack(flows)
        flows = rearrange(flows, "t c h w -> t h w c")
        return flows

    def load_frames(self, frames):
        # Resize all images to original_image_shape
        H, W = self.cfg.original_image_shape
        resized_images = []
        for frame in frames:
            file_path = frame["rgb_file_path"].replace("images", "images_8")
            img = Image.open(file_path).convert("RGB")
            if img.size != (W, H):
                img = img.resize((W, H), Image.BILINEAR)
            resized_images.append(self.to_tensor(img))

        torch_images = torch.stack(resized_images)
        return torch_images

    def load_masks(self, scene, mask_root, threhold: float = 0.8):
        mask_file = Path(mask_root) / f"{scene}_masks.npy"
        if not mask_file.exists():
            raise FileNotFoundError(f"Mask file not found at {mask_file}")

        masks = np.load(str(mask_file))
        masks = torch.tensor(masks, dtype=torch.float32)
        masks = (masks > threhold).float()
        return masks
        
    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]
        
    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:
        scene = self.scene_ids[index]
        
        example = self.scenes[scene]
        # load poses
        extrinsics = []
        intrinsics = []
        for frame in example:
            extrinsic = frame["extrinsics"]
            intrinsic = frame["intrinsics"]
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        
        extrinsics = np.array(extrinsics)
        intrinsics = np.array(intrinsics)
        extrinsics = np.linalg.inv(extrinsics)
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

        intrinsic = [[0.4836, 0.0000, 0.5000],[0.0000, 0.8597, 0.5000],[0.0000, 0.0000, 1.0000]] # re10K intrinsic
        intrinsics = torch.tensor(intrinsic, dtype=torch.float32).unsqueeze(0).repeat(intrinsics.shape[0], 1, 1)
        
        # Calculate masks from flows and depths
        # flows = self.load_flows(example)
        # flows = flows[:, ::2, ::2] / 2
        # flows = torch.nan_to_num(flows, nan=0.0)
        # depths = self.load_depths(example, intrinsics[0])
        # depths = depths[:, ::2, ::2]
        # flows[..., 0] *= (patchsize[1] * 14) / flows.shape[2]
        # flows[..., 1] *= (patchsize[0] * 14) / flows.shape[1]
        # flows = F.interpolate(
        #     flows.permute(0, 3, 1, 2),
        #     size=(patchsize[0] * 14, patchsize[1] * 14),
        #     mode="bilinear",
        #     align_corners=False,
        # ).permute(0, 2, 3, 1)
        # depths = F.interpolate(
        #     depths.unsqueeze(1),
        #     size=(patchsize[0] * 14, patchsize[1] * 14),
        #     mode="bilinear",
        #     align_corners=False,
        # ).squeeze(1)
        # masks = flow_to_dynamic_mask(flows, depths, extrinsics, intrinsics)
        # last_frame_mask = torch.full_like(masks[:1], -1.0)
        # masks = torch.cat([masks, last_frame_mask], dim=0)  # [T-1, H, W] -> [T, H, W]

        # Load preprocessed masks
        masks = self.load_masks(scene, self.cfg.mask_root)

        try:
            context_indices, target_indices, overlap = self.view_sampler.sample(
                scene,
                # num_context_views,
                extrinsics,
                intrinsics,
            )
        except ValueError:
            # Skip because the example doesn't have enough frames.
            raise Exception("Not enough frames")
        
        # Skip the example if the field of view is too wide.
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise Exception("Field of view too wide")
        
        # Load the images.
        input_frames = [example[i] for i in context_indices]
        target_frame = [example[i] for i in target_indices]
        
        context_images = self.load_frames(input_frames)
        target_images = self.load_frames(target_frame)
        
        # Skip the example if the images don't have the right shape.
        context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
        target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
        if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
            raise Exception("Bad example image shape")
        
        # Resize the world to make the baseline 1.
        context_extrinsics = extrinsics[context_indices]
        if self.cfg.make_baseline_1:
            a = context_extrinsics[0, :3, 3]              # 기준(0번) 위치
            b_all = context_extrinsics[1:, :3, 3]         # 나머지 모든 extrinsics 위치들

            diff = b_all - a                              # shape: (N-1, 3)
            norms = diff.norm(dim=1)                      # 각 거리의 norm
            scale = norms.max()                           # 그 중 max
            if scale < self.cfg.baseline_min:
                # Check if ALL cameras (context + target) are perfectly static
                all_positions = extrinsics[:, :3, 3]
                all_diffs = (all_positions - all_positions[0:1]).norm(dim=1)
                if all_diffs.max() < self.cfg.baseline_min:
                    # Static camera — assign identity extrinsics to all
                    extrinsics = torch.eye(4, dtype=extrinsics.dtype).unsqueeze(0).expand(extrinsics.shape[0], -1, -1).clone()
                    scale = 1
                else:
                    print(
                        f"Skipped {scene} because of baseline out of range: "
                        f"{scale:.6f}"
                    )
                    raise Exception("baseline out of range")
            elif scale > self.cfg.baseline_max:
                print(
                    f"Skipped {scene} because of baseline out of range: "
                    f"{scale:.6f}"
                )
                raise Exception("baseline out of range")
            else:
                extrinsics[:, :3, 3] /= scale
        else:
            scale = 1
        
        if self.cfg.relative_pose:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            raise Exception("encounter nan or inf in input poses")
    
        context_masks = masks[context_indices]
        target_masks = masks[target_indices]

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
                "camera": torch.zeros_like(context_indices),  # dummy
                "mask": context_masks,
                # "overlap": overlap,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
                "camera": torch.zeros_like(target_indices),  # dummy
                "mask": target_masks,
            },
            "scene": "spring_"+scene,
            "dataset_name": self.cfg.name,
        }
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)
        
        example = apply_crop_shim(example, (patchsize[0] * 14, patchsize[1] * 14))
        return example
        
    def __getitem__(self, index: int) -> dict:
        num_context_views = self.view_sampler.num_context_views
        patchsize_h, patchsize_w = self.cfg.input_image_shape
        patchsize_h = patchsize_h // 14
        patchsize_w = patchsize_w // 14
        try:
            return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))
        except Exception as e:
            print(f"Error: {e}")
            # traceback.print_exc()
            index = np.random.randint(len(self))
            return self.__getitem__(index)
    
    def _read_frame(self, video_path: str, index: int) -> np.ndarray:
        if iio is None:
            raise ImportError("imageio.v3 is required to read video frames on-the-fly.")
        mock.patch("imageio_ffmpeg._io.subprocess.Popen.kill").start()
        frame = iio.imread(video_path, index=index)  # H W 3 (uint8)
        # Resize to original_image_shape (H, W)
        H, W = self.cfg.original_image_shape
        img = Image.fromarray(frame)
        if img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)
        return np.asarray(img)

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

    def __len__(self) -> int:
        return len(self.scene_ids)
