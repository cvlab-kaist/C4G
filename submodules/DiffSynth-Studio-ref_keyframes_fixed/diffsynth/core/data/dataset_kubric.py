"""
Kubric dataset for Wan2.1 VACE deblur training.

Blur root layout:
  {blur_root}/{scene_type}/{scene_num}/view_{ref:04d}_{rendered:04d}/{start}_{end}_{gap}_{t}.png

Context root layout:
  {context_root}/{scene_type}/frames/{scene_num}/view_{cam:04d}/rgba_{i:05d}.png
"""

import os
import json
from dataclasses import dataclass

import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
import sys

BLUR_ROOT = "/path/to/kubric_blur_root"
CONTEXT_ROOT = "/path/to/kubric"

@dataclass
class DatasetSpringDeblurCfg:
    blur_root:    str = BLUR_ROOT
    context_root: str = CONTEXT_ROOT
    image_height: int = 480
    image_width:  int = 832
    caption_json: str | None = None
    

class DatasetSpringDeblur(Dataset):
    """
    One sample = (scene, gap, ctx_ctr).

    context_start   = ctx_ctr * gap
    context_indices = [context_start + i*gap for i in range(12)]
    targets         = [context_start + tgt_ctr*gap + gap//2 for tgt_ctr in range(11)]
    """

    def __init__(self, cfg: DatasetSpringDeblurCfg):
        self.cfg       = cfg
        self.to_tensor = tf.ToTensor()
        self.load_from_cache = False

        self._scan_samples(cfg)

        cap_path = cfg.caption_json or os.path.join(cfg.blur_root, "captions.json")
        self.captions: dict[str, str] = {}
        if os.path.exists(cap_path):
            with open(cap_path, "r", encoding="utf-8") as f:
                self.captions = json.load(f)
            print(f"DatasetKubricDeblur: loaded {len(self.captions)} captions from {cap_path}")
        else:
            print(f"DatasetKubricDeblur: no caption file at {cap_path} — caption will be empty string")

    def _scan_samples(self, cfg):
        """Scan all valid sample windows from blur_root."""
        windows: dict[tuple, set] = {}
        for scene_type in sorted(os.listdir(cfg.blur_root)):
            scene_type_dir = os.path.join(cfg.blur_root, scene_type)
            if not os.path.isdir(scene_type_dir):
                continue
            for scene_num in sorted(os.listdir(scene_type_dir)):
                scene_num_dir = os.path.join(scene_type_dir, scene_num)
                if not os.path.isdir(scene_num_dir):
                    continue
                for view_folder in sorted(os.listdir(scene_num_dir)):
                    if not view_folder.startswith("view_"):
                        continue
                    view_parts = view_folder.split("_")
                    if len(view_parts) != 3:
                        continue
                    try:
                        ref_cam = int(view_parts[1])
                        rendered_cam = int(view_parts[2])
                    except ValueError:
                        continue
                    view_dir = os.path.join(scene_num_dir, view_folder)
                    if not os.path.isdir(view_dir):
                        continue
                    for fname in sorted(os.listdir(view_dir)):
                        if not fname.endswith(".png"):
                            continue
                        parts = fname[:-4].split("_")
                        if len(parts) != 4:
                            continue
                        try:
                            start, end, gap, t = map(int, parts)
                        except ValueError:
                            continue
                        key = (scene_type, scene_num, ref_cam, rendered_cam, start, end, gap)
                        windows.setdefault(key, set()).add(t)

        self.samples: list[tuple] = []
        skipped = 0
        for key, found in windows.items():
            _, _, _, _, start, end, _ = key
            if set(range(start, end + 1)) == found:
                self.samples.append(key)
            else:
                skipped += 1
        print(f"DatasetKubricDeblur: {len(self.samples)} samples ({skipped} skipped)")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _pad_to_4k1(video: torch.Tensor, idx: torch.Tensor):
        """Append last frame copies until length becomes 4k+1 (CausalConv3d clean round-trip)."""
        T = video.shape[0]
        if (T - 1) % 4 == 0:
            return video, idx
        target = ((T - 1) // 4 + 1) * 4 + 1
        pad_n = target - T
        video = torch.cat([video, video[-1:].repeat(pad_n, *([1] * (video.dim() - 1)))], dim=0)
        idx   = torch.cat([idx,   idx[-1:].repeat(pad_n)], dim=0)
        return video, idx

    def __getitem__(self, index: int) -> dict:
        try:
            return self._getitem_impl(index)
        except FileNotFoundError as e:
            next_index = (index + 1) % len(self.samples)
            print(f"[DatasetKubricDeblur] SKIP idx={index} (missing file: {e.filename}) → try idx={next_index}", flush=True)
            return self.__getitem__(next_index)

    def _getitem_impl(self, index: int) -> dict:
        scene_type, scene_num, ref_cam, rendered_cam, start, end, gap = self.samples[index]
        scene_key = f"{scene_type}/frames/{scene_num}"   # subpath under context_root
        # - target (view2): 23 dense frames  -> start..start+22
        # - context(view1): 12 keyframes     -> start,start+2,...,start+22
        target_end = start + 22
        keyframe_indices = list(range(start, target_end + 1, 2))
        all_frames = list(range(start, target_end + 1))

        # ref_cam provides clean keyframes (context), rendered_cam is the blur-render target.
        view1 = f"view_{ref_cam:04d}"       # context_root folder name
        view2 = f"view_{rendered_cam:04d}"  # context_root folder name (clean GT for rendered_cam)

        rendered_video = torch.stack(
            [self._load_rendered(scene_type, scene_num, ref_cam, rendered_cam, start, end, gap, t)
             for t in all_frames]
        )  # (N, C, H, W)
        rendered_video_idx = torch.tensor(all_frames, dtype=torch.long)

        # gt_video_view1: ref_cam clean context keyframes
        gt_video_view1 = torch.stack(
            [self._load_clean(scene_key, view1, t)[0] for t in keyframe_indices]
        )  # (num_keyframes, C, H, W)
        gt_video_idx1 = torch.tensor(keyframe_indices, dtype=torch.long)

        # gt_video_view2: rendered_cam clean GT (all frames)
        gt_video_view2 = torch.stack(
            [self._load_clean(scene_key, view2, t)[0] for t in all_frames]
        )  # (N, C, H, W)
        gt_video_idx2 = torch.tensor(all_frames, dtype=torch.long)

        # Save original indices (before padding)
        context_idx_orig = gt_video_idx1.clone()    # keyframe pixel timestamps
        target_idx_orig  = gt_video_idx2.clone()    # rendered pixel timestamps

        # 4k+1 padding: fill missing frames by repeating the last frame (CausalConv3d constraint).
        # view1 (context keyframes) is encoded frame-by-frame, so no padding is needed.
        gt_video_view2, gt_video_idx2     = self._pad_to_4k1(gt_video_view2, gt_video_idx2)
        rendered_video, rendered_video_idx = self._pad_to_4k1(rendered_video, rendered_video_idx)

        gt_video = torch.cat([gt_video_view1, gt_video_view2], dim=0)

        # context_video = keyframes (no padding; view1 is processed frame-by-frame by VAE)
        context_video = gt_video_view1
        context_video_keyframes = gt_video_view1

        vace_video = torch.cat([context_video, rendered_video], dim=0)
        # RoPE: ref(view1) tokens get +500 offset so their positions don't collide with rendered
        REF_POS_OFFSET = 500
        vace_video_idx = torch.cat([gt_video_idx1 + REF_POS_OFFSET, rendered_video_idx], dim=0)
        vace_mask = torch.cat([torch.zeros(len(context_video), dtype=torch.float32), torch.ones(len(rendered_video), dtype=torch.float32)], dim=0)

        caption = self.captions.get(f"{scene_type}/{scene_num}", "")

        return {
            "scene"                  : f"{scene_key}",
            "view"                   : {"context": view1, "blur": view2},
            "caption"                : caption,
            "gt_video"               : gt_video,
            "vace_video"             : vace_video,
            "vace_video_idx"         : vace_video_idx,
            "vace_video_mask"        : vace_mask,
            "context_video_keyframes": context_video_keyframes,
            "context_idx_orig"       : context_idx_orig,  # keyframe timestamps (without padding)
            "target_idx_orig"        : target_idx_orig,   # rendered/target timestamps (without padding)
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_rendered(self, scene_type, scene_num, ref_cam, rendered_cam, start, end, gap, t) -> torch.Tensor:
        # {blur_root}/{scene_type}/{scene_num}/view_{ref:04d}_{rendered:04d}/{start}_{end}_{gap}_{t}.png
        path = os.path.join(
            self.cfg.blur_root, scene_type, scene_num,
            f"view_{ref_cam:04d}_{rendered_cam:04d}",
            f"{start}_{end}_{gap}_{t}.png",
        )
        return self._load_image(path)

    def _load_clean(self, scene_key, view, frame_idx) -> tuple:
        # {context_root}/{scene_key}/{view}/rgba_{frame_idx:05d}.png
        path = os.path.join(self.cfg.context_root, scene_key, view,
                            f"rgba_{frame_idx:05d}.png")
        return self._load_image(path), frame_idx

    def _load_image(self, path: str) -> torch.Tensor:
        # Anisotropic resize (no crop) to match the splat decoder, which renders
        # K-calibrated 1:1 FOV onto a non-square viewport by stretching.
        H, W = self.cfg.image_height, self.cfg.image_width
        img = Image.open(path).convert("RGB")
        tensor = self.to_tensor(img)   # (C, H_orig, W_orig)
        _, h_in, w_in = tensor.shape
        if h_in != H or w_in != W:
            tensor = TF.resize(tensor, (H, W), antialias=True)
        return tensor
