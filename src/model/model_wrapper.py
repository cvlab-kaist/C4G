from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol, runtime_checkable, Any, Tuple
from itertools import accumulate

import moviepy.editor as mpy
import torch
import torch.nn.functional as F
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from .encoder.common.gaussians import build_covariance
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..geometry.rotations import slerp_batch
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_ss import ScaleAndShiftInvariantLoss, normal_map_loss
from ..geometry.ptc_geometry import depthmap_to_pts3d
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, camera_normalization
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from .types import Gaussians
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.flows import flow_uv_to_colors
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..loss.loss_tracking import TrackingConsistencyLoss, project_points_to_image
import utils3d  # git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    visualize_gaussian_token: int = -1
    forward_vfm: bool = False
    labels: list[str] = field(default_factory=lambda: ['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other'])
    color_hex_list: list[str] = field(default_factory=lambda: ['#000000', '#E6194B','#3CB44B','#FFE119','#4363D8','#F58231','#911EB4','#42D4F4','#808000'])

@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    reproj_model: str = 'vggt' # 'vggt' or 'moge'
    depth_loss: float = 0.0
    normal_loss: float = 0.0
    depth_loss_step: int = 0
    tracking_consistency_loss: float = 0.0
    render_time_interpolation: bool = False
    render_time_interpolation_colored: bool = False
    validation_visualization: bool = True
    validation_attention_visualization: bool = True
    validation_lpips_batch_size: int = 1


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass
    
class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    # Frozen / optional modules whose weights should never be saved or
    # loaded from our own checkpoints (they load their own pretrained weights).
    _EXCLUDED_PREFIXES = ("tracking_loss_fn.", "vggt.", "depth_loss_func.")

    def state_dict(self, *args, **kwargs):
        """Exclude frozen / optional module weights from checkpoints."""
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items()
                if not any(k.startswith(p) for p in self._EXCLUDED_PREFIXES)}

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Filter out frozen / optional module weights from checkpoint.
        These modules are frozen and load their own pretrained weights,
        so checkpoint keys are never needed."""
        state_dict = checkpoint.get("state_dict", {})
        keys_to_remove = [k for k in state_dict
                          if any(k.startswith(p) for p in self._EXCLUDED_PREFIXES)]
        for k in keys_to_remove:
            del state_dict[k]

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Temporarily unregister frozen / optional modules so their keys
        are excluded from the strict check on both sides (checkpoint and model)."""
        saved = {}
        for attr in ("tracking_loss_fn", "vggt", "depth_loss_func"):
            if hasattr(self, attr):
                saved[attr] = getattr(self, attr)
                setattr(self, attr, None)
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        for attr, module in saved.items():
            setattr(self, attr, module)
        return result

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        vggt = None,
        mode: str = "train"
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self._data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        self.mode=mode

        # Only register vggt / depth_loss_func during training.
        # They are frozen pretrained models never used at eval time, and
        # registering them as submodules causes checkpoint key mismatches.
        if mode == "train" and vggt is not None:
            self.vggt = vggt
        else:
            self.vggt = None

        if mode == "train" and (self.train_cfg.depth_loss > 0 or self.train_cfg.normal_loss > 0):
            self.depth_loss_func = ScaleAndShiftInvariantLoss()

        # Accumulate per-dataset validation metrics for epoch-end averaging.
        self._val_metrics: dict[str, list[dict[str, float]]] = {}

        # Tracking loss is only needed during training.
        self.tracking_loss_fn = None
        if self.mode == "train":
            if self.train_cfg.tracking_consistency_loss > 0:
                self.tracking_loss_fn = TrackingConsistencyLoss(
                    cowtracker_path=None,  # Will auto-download from HuggingFace
                    num_query_points=256,
                )

    def apply_data_shim(self, batch: Dict[str, Any]) -> BatchedExample:
        return self._data_shim(batch)

    def training_step(self, multi_batch, batch_idx):
        multi_total_loss = 0

        if not isinstance(multi_batch, list):
            multi_batch = [multi_batch]

        for batch in multi_batch:

            # combine batch from different dataloaders
            if isinstance(batch, list):
                batch_combined = None
                for batch_per_dl in batch:
                    if batch_combined is None:
                        batch_combined = batch_per_dl
                    else:
                        for k in batch_combined.keys():
                            if isinstance(batch_combined[k], list):
                                batch_combined[k] += batch_per_dl[k]
                            elif isinstance(batch_combined[k], dict):
                                for kk in batch_combined[k].keys():
                                    batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                            else:
                                raise NotImplementedError
                batch = batch_combined

            batch: BatchedExample = self.apply_data_shim(batch)

            b, _, _, h, w = batch["target"]["image"].shape
            all_timestamps = torch.cat([batch['context']['index'], batch['target']['index']], dim=1)

            visualization_dump = {}

            total_loss = 0
            unique_timestamps = torch.unique(all_timestamps)
            unique_timestamps_list = unique_timestamps.tolist()

            gaussians_per_timestamp = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump, target_timestamps=unique_timestamps)

            for t_idx in unique_timestamps_list:
                gaussians_full = gaussians_per_timestamp[t_idx]
                context_mask = (batch["context"]["index"] == t_idx)
                target_mask = (batch["target"]["index"] == t_idx)

                for b_idx in range(b):
                    ctx_mask_b = context_mask[b_idx]
                    tgt_mask_b = target_mask[b_idx]
                    if not (ctx_mask_b.any() or tgt_mask_b.any()):
                        continue

                    gaussians = Gaussians(
                        gaussians_full.means[b_idx:b_idx+1],
                        gaussians_full.covariances[b_idx:b_idx+1],
                        gaussians_full.harmonics[b_idx:b_idx+1],
                        gaussians_full.opacities[b_idx:b_idx+1],
                    )

                    n_ctx_views = ctx_mask_b.sum().item()
                    n_tgt_views = tgt_mask_b.sum().item()

                    output = self.decoder.forward(
                        gaussians,
                        torch.cat([batch["context"]["extrinsics"][b_idx, ctx_mask_b], batch["target"]["extrinsics"][b_idx, tgt_mask_b]], dim=0).unsqueeze(0),
                        torch.cat([batch["context"]["intrinsics"][b_idx, ctx_mask_b], batch["target"]["intrinsics"][b_idx, tgt_mask_b]], dim=0).unsqueeze(0),
                        torch.cat([batch["context"]["near"][b_idx, ctx_mask_b], batch["target"]["near"][b_idx, tgt_mask_b]], dim=0).unsqueeze(0),
                        torch.cat([batch["context"]["far"][b_idx, ctx_mask_b], batch["target"]["far"][b_idx, tgt_mask_b]], dim=0).unsqueeze(0),
                        (h, w),
                        depth_mode=self.train_cfg.depth_mode,
                        global_step=self.global_step,
                    )

                    context_gt = ((batch["context"]["image"][b_idx, ctx_mask_b] + 1) / 2)
                    target_gt = batch["target"]["image"][b_idx, tgt_mask_b]
                    target_gt = torch.cat([context_gt, target_gt], dim=0).unsqueeze(0)

                    # Compute metrics.
                    psnr_probabilistic = compute_psnr(
                        rearrange(target_gt, "b v c h w -> (b v) c h w"),
                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                    )
                    self.log("train/psnr_probabilistic", psnr_probabilistic.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
                    if "dataset_name" in batch:
                        ds_name = batch["dataset_name"][0] if isinstance(batch["dataset_name"], list) else batch["dataset_name"]
                        self.log(f"train/psnr_{ds_name}", psnr_probabilistic.mean(), on_step=True, on_epoch=True, logger=True)

                    # Compute and log loss.
                    context_dynamic_mask = batch["context"].get("mask", None)
                    target_dynamic_mask = batch["target"].get("mask", None)
                    if context_dynamic_mask is not None and target_dynamic_mask is not None:
                        context_dynamic_mask = context_dynamic_mask[b_idx, ctx_mask_b]
                        target_dynamic_mask = target_dynamic_mask[b_idx, tgt_mask_b]
                        dynamic_mask = torch.cat([context_dynamic_mask, target_dynamic_mask], dim=0)
                    else:
                        dynamic_mask = None

                    for loss_fn in self.losses:
                        if getattr(loss_fn, 'has_dynamic_mask', False):
                            loss = loss_fn.forward(output, batch, gaussians, self.global_step, target_image=target_gt, dynamic_mask=dynamic_mask)
                        else:
                            loss = loss_fn.forward(output, batch, gaussians, self.global_step, target_image=target_gt)
                        self.log(f"loss/{loss_fn.name}", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                        total_loss = total_loss + loss

                    if (self.train_cfg.depth_loss > 0 or self.train_cfg.normal_loss > 0) and self.global_step > self.train_cfg.depth_loss_step:
                        pseudo_gt_depth, pseudo_gt_normal = self.get_depth_pseudo_labels(target_gt) # (b v) h w, (b v) h w c
                        pseudo_gt_depth[pseudo_gt_depth.isinf()] = 0

                        depth_loss, _, _ = self.depth_loss_func(rearrange(output.depth, "b v h w -> (b v) h w"), pseudo_gt_depth.detach(), mask = (pseudo_gt_depth.detach()>0))

                        self.log("loss/depth_loss", depth_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                        total_loss = total_loss + self.train_cfg.depth_loss * depth_loss

                        if self.train_cfg.normal_loss > 0:
                            pred_normal = self.convert_depth_to_normal(rearrange(output.depth, "b v h w -> (b v) h w"), torch.cat([batch["context"]["intrinsics"][b_idx, ctx_mask_b], batch["target"]["intrinsics"][b_idx, tgt_mask_b]], dim=0)) # (b v) c h w
                            normal_loss, _ = normal_map_loss(pred_normal, pseudo_gt_normal.detach())
                            self.log("loss/normal_loss", normal_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                            total_loss = total_loss + self.train_cfg.normal_loss * normal_loss

                            del pseudo_gt_normal, pred_normal
                        del pseudo_gt_depth
                        del output

            tracking_weight = self.train_cfg.tracking_consistency_loss
            if (tracking_weight > 0
                and self.tracking_loss_fn is not None
                and len(gaussians_per_timestamp) > 1):
                try:
                    # Get camera IDs if available, otherwise use indices
                    context_cameras = batch["context"].get("camera", None)
                    target_cameras = batch["target"].get("camera", None)

                    tracking_loss = self.tracking_loss_fn(
                        gaussians_per_timestamp=gaussians_per_timestamp,  # {timestamp: [B, num_gaussians, 3]}
                        context_images=batch["context"]["image"],  # [B, num_context, 3, H, W]
                        context_extrinsics=batch["context"]["extrinsics"],  # [B, num_context, 4, 4]
                        context_intrinsics=batch["context"]["intrinsics"],  # [B, num_context, 3, 3]
                        context_timestamps=batch["context"]["index"],  # [B, num_context]
                        context_cameras=context_cameras,  # [B, num_context] or None
                        target_images=batch["target"]["image"],  # [B, num_target, 3, H, W]
                        target_extrinsics=batch["target"]["extrinsics"],  # [B, num_target, 4, 4]
                        target_intrinsics=batch["target"]["intrinsics"],  # [B, num_target, 3, 3]
                        target_timestamps=batch["target"]["index"],  # [B, num_target]
                        target_cameras=target_cameras,  # [B, num_target] or None
                    )
                    self.log("loss/tracking_consistency", tracking_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                    total_loss = total_loss + tracking_weight * tracking_loss
                except Exception as e:
                    print(f"Warning: Tracking consistency loss failed: {e}")
                    raise e

            total_loss = total_loss / len(unique_timestamps_list)

            self.log("loss/total", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            if (
                self.global_rank == 0
                and self.global_step % self.train_cfg.print_log_every_n_steps == 0
                and (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0
            ):
                print(
                    f"train step {self.global_step}; "
                    f"scene = {[x[:20] for x in batch['scene']]}; "
                    f"context = {batch['context']['index'].tolist()}; "
                    f"loss = {total_loss:.6f}",
                    f"low_pass_filter = {self.decoder.low_pass_filter:.3f}"
                )
            multi_total_loss = multi_total_loss + total_loss
        multi_total_loss = multi_total_loss / len(multi_batch)
        self.log("loss/multi_total_loss", multi_total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("info/global_step", self.global_step, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return multi_total_loss

    def test_step(self, batch, batch_idx):
        """
        C4G Test Step: 각 target timestamp에 대해 encoder를 호출하여 Gaussian 생성 후 렌더링.
        
        C3G와의 차이점:
        - encoder에 target_timestamp 전달 필요
        - 각 timestamp마다 별도로 encoder 호출
        - tto_gs 미사용
        """
        batch: BatchedExample = self.apply_data_shim(batch)

        b, v, _, h, w = batch["target"]["image"].shape
    
        if self.global_rank == 0:
            print(
                f"test step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"target = {batch['target']['index'].tolist()}; "
                f"target_camera = {batch['target']['camera'].tolist()}"
            )
        
        # Resize context images if needed
        if h != 224 or w != 224:
            b, cv, _, ch, cw = batch['context']['image'].shape
            batch['context']['image'] = F.interpolate(
                batch['context']['image'].reshape(b * cv, 3, ch, cw), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).reshape(b, cv, 3, 224, 224)
        
        assert b == 1
        # if batch_idx % 100 == 0:
        print(f"Test step {batch_idx:0>6}.")

        # Setup paths
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        os.makedirs(path / scene / "color", exist_ok=True)

        # Get all unique target timestamps
        all_timestamps = torch.unique(batch['target']['index'])
        
        # Collect results across all timestamps
        all_rgb_pred = []
        all_rgb_gt = []
        all_depth_pred = []
        all_gaussians = []
        
        # Encode with target timestamp
        with self.benchmarker.time("encoder"):
            gaussian_per_timestamp = self.encoder(
                batch["context"],
                self.global_step,
                target_timestamps=torch.unique(all_timestamps),
            )
            
        for t_idx in sorted(all_timestamps.tolist()):
            # Get mask for current timestamp
            gaussians = gaussian_per_timestamp[t_idx]
            target_mask = (batch["target"]["index"] == t_idx).squeeze(0)  # [v]
            
            all_gaussians.append(gaussians)
            
            # Decode for target views at this timestamp
            target_extrinsics = batch["target"]["extrinsics"][:, target_mask]
            target_intrinsics = batch["target"]["intrinsics"][:, target_mask]
            target_near = batch["target"]["near"][:, target_mask]
            target_far = batch["target"]["far"][:, target_mask]
            num_target_views = target_extrinsics.shape[1]
            
            if self.test_cfg.align_pose:
                output = self.test_step_align_single_timestamp(
                    batch, gaussians, target_mask, h, w
                )
            else:
                with self.benchmarker.time("decoder", num_calls=num_target_views):
                    output = self.decoder.forward(
                        gaussians,
                        target_extrinsics,
                        target_intrinsics,
                        target_near,
                        target_far,
                        (h, w),
                    )
            
            # Collect predictions and ground truth
            all_rgb_pred.append(output.color[0])  # [num_cams_at_t, 3, h, w]
            all_rgb_gt.append(batch["target"]["image"][:, target_mask].squeeze(0))  # [num_cams_at_t, 3, h, w]
            all_depth_pred.append(vis_depth_map(output.depth[0]))
            
            # Save images per timestamp
            if self.test_cfg.save_image:
                target_cameras = batch["target"]["camera"][:, target_mask].squeeze(0)
                for cam_id, color in zip(target_cameras, output.color[0]):
                    save_image(color, path / scene / f"color/t{t_idx:04d}_cam{cam_id.item():02d}.png")

        # Concatenate all results
        rgb_pred = torch.cat(all_rgb_pred, dim=0)
        rgb_gt = torch.cat(all_rgb_gt, dim=0)
        depth_pred = torch.cat(all_depth_pred, dim=0)

        # Compute scores
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)
            
            psnr = compute_psnr(rgb_gt, rgb_pred).mean()
            all_metrics = {
                "lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                "ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                "psnr_ours": psnr,
                "num_gaussians_ours": all_gaussians[0].means.shape[1],
            }
            methods = ['ours']

            self.log_dict(all_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)

        # Save context images
        if self.test_cfg.save_image:
            context_img = inverse_normalize(batch["context"]["image"][0])
            for ctx_idx, ctx_img in enumerate(context_img):
                ctx_t = batch["context"]["index"][0, ctx_idx].item()
                ctx_cam = batch["context"]["camera"][0, ctx_idx].item()
                save_image(ctx_img, path / scene / f"context_t{ctx_t:04d}_cam{ctx_cam:02d}.png")

        # Save video (frames ordered by timestamp)
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in rgb_pred],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        # Save Gaussian projections (use last timestamp's gaussians)
        projections = hcat(
            *render_projections(
                all_gaussians[-1],
                256,
                extra_label="",
                low_pass=self.decoder.low_pass_filter,
            )[0]
        )
        save_image(projections, path / f"{scene}_projections.png")

        # Save comparison image
        if self.test_cfg.save_compare:
            context_img = inverse_normalize(batch["context"]["image"][0])
            error_map = (rgb_gt - rgb_pred.clamp(0, 1)).abs()
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
                add_label(vcat(*error_map), "Error Map"),
            )
            save_image(comparison, path / f"{scene}_{psnr:.3f}.png")
            
            # Save depth
            save_image(hcat(*depth_pred), path / f"{scene}_depth.png")

    def test_step_align_single_timestamp(self, batch, gaussians, target_mask, h, w):
        """Align pose for a single timestamp's target views."""
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        target_extrinsics = batch["target"]["extrinsics"][:, target_mask]
        target_intrinsics = batch["target"]["intrinsics"][:, target_mask]
        target_near = batch["target"]["near"][:, target_mask]
        target_far = batch["target"]["far"][:, target_mask]
        target_image = batch["target"]["image"][:, target_mask]
        
        b = target_extrinsics.shape[0]
        v = target_extrinsics.shape[1]

        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = [
                {"params": [cam_rot_delta], "lr": self.test_cfg.rot_opt_lr},
                {"params": [cam_trans_delta], "lr": self.test_cfg.trans_opt_lr},
            ]
            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = target_extrinsics.clone()
            
            prev_loss = None
            patience_counter = 0
            patience_limit = 10

            with self.benchmarker.time("optimize"):
                for i in tqdm(range(self.test_cfg.pose_align_steps * 10), desc="Pose align"):
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        target_intrinsics,
                        target_near,
                        target_far,
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                    )

                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step, target_image=target_image)
                        total_loss = total_loss + loss

                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(
                            cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                            cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                            extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
                        )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)
                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

                    if prev_loss is not None:
                        delta = abs(total_loss.item() - prev_loss)
                        if delta < 0.00001:
                            patience_counter += 1
                            if patience_counter >= patience_limit and i >= 100:
                                break
                        else:
                            patience_counter = 0
                    prev_loss = total_loss.item()

        output = self.decoder.forward(
            gaussians,
            extrinsics,
            target_intrinsics,
            target_near,
            target_far,
            (h, w),
        )
        del pose_optimizer
        return output
 
    # image-level iou and acc
    def on_test_epoch_end(self):
        mean_iou = sum(self.per_image_ious) / len(self.per_image_ious) if self.per_image_ious else 0.0
        mean_acc = sum(self.per_image_accs) / len(self.per_image_accs) if self.per_image_accs else 0.0

        print("mIoU:", mean_iou)
        print("Acc:", mean_acc)

        self.log("test/mIoU", mean_iou, prog_bar=True)
        self.log("test/Acc", mean_acc, prog_bar=True)

        # Reset lists for next epoch
        self.per_image_ious.clear()
        self.per_image_accs.clear()

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined

        batch: BatchedExample = self.apply_data_shim(batch)
        b, _, _, h, w = batch["target"]["image"].shape
        # all_timestamps = batch['target']['index']
        all_timestamps = torch.cat([batch['context']['index'], batch['target']['index']], dim=1)
        num_timestamps = all_timestamps.shape[1]

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"context camera = {batch['context']['camera'].tolist()}; "
                f"target = {batch['target']['index'].tolist()}; "
                f"target_camera = {batch['target']['camera'].tolist()}"
            )

        # Run the model.
        do_visualization = self.train_cfg.validation_visualization
        do_attention_visualization = (
            do_visualization and self.train_cfg.validation_attention_visualization
        )
        visualization_dump = {} if do_attention_visualization else None

        total_loss = 0
        num_cameras = 3
        rgb_pred, depth_pred = [], []
        gs_list = []
        rgb_gt = []
        torch.cuda.synchronize()
        _t0 = torch.cuda.Event(enable_timing=True)
        _t1 = torch.cuda.Event(enable_timing=True)
        _t0.record()
        gaussian_per_timestamp = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump, target_timestamps=torch.unique(all_timestamps))
        _t1.record()
        torch.cuda.synchronize()
        _encoder_ms = _t0.elapsed_time(_t1)

        _t0.record()
        for t_idx in sorted(torch.unique(all_timestamps).tolist()):
            gaussians = gaussian_per_timestamp[t_idx]
            context_mask = (batch["context"]["index"] == t_idx)
            target_mask = (batch["target"]["index"] == t_idx)

            output = self.decoder.forward(
                gaussians,
                torch.cat([batch["context"]["extrinsics"][context_mask], batch["target"]["extrinsics"][target_mask]], dim=0).unsqueeze(0),
                torch.cat([batch["context"]["intrinsics"][context_mask], batch["target"]["intrinsics"][target_mask]], dim=0).unsqueeze(0),
                torch.cat([batch["context"]["near"][context_mask], batch["target"]["near"][target_mask]], dim=0).unsqueeze(0),
                torch.cat([batch["context"]["far"][context_mask], batch["target"]["far"][target_mask]], dim=0).unsqueeze(0),
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
                global_step=self.global_step,
            )

            rgb_pred.append(output.color[0])

            rgb_gt.append(torch.cat([((batch["context"]["image"][context_mask] + 1) / 2), batch["target"]["image"][target_mask]], dim=0))
            if do_visualization:
                depth_pred.append(vis_depth_map(output.depth[0]))
            gs_list.append(gaussians)

        _t1.record()
        torch.cuda.synchronize()
        _decoder_ms = _t0.elapsed_time(_t1)

        if self.global_rank == 0:
            print(
                f"val step {self.global_step}; "
                f"encoder = {_encoder_ms:.1f}ms; "
                f"decoder = {_decoder_ms:.1f}ms"
            )

        if do_visualization and self.train_cfg.render_time_interpolation:
            self.render_time_interpolation(
                batch,
                gs_list
            )

        if do_visualization and self.train_cfg.render_time_interpolation_colored:
            self.render_time_interpolation_colored(
                batch,
                gs_list
            )

        if do_visualization:
            self.render_all_timestamps(
                batch
            )

            self.render_all_timestamps_colored(
                batch,
                gs_list
            )

            self.render_interpolated_timeline(
                batch,
                gaussian_per_timestamp,
            )

            self.visualize_gaussian_center_tracks(
                batch,
                gaussian_per_timestamp,
            )

            self.visualize_gaussian_flow_map(
                batch,
                gaussian_per_timestamp,
            )

        rgb_gt = torch.cat(rgb_gt, dim=0)
        rgb_pred = torch.cat(rgb_pred, dim=0)

        # Compute validation metrics.
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        lpips = self.compute_lpips_chunked(rgb_gt, rgb_pred).mean()
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()

        # Per-dataset logging
        ds_name = ""
        if "dataset_name" in batch:
            ds_name = batch["dataset_name"][0] if isinstance(batch["dataset_name"], list) else batch["dataset_name"]
        if ds_name:
            self.log(f"val/psnr_{ds_name}", psnr)
            self.log(f"val/lpips_{ds_name}", lpips)
            self.log(f"val/ssim_{ds_name}", ssim)
            self._val_metrics.setdefault(ds_name, []).append({
                "psnr": psnr.item(), "lpips": lpips.item(), "ssim": ssim.item(),
            })
        else:
            self.log("val/psnr", psnr)
            self.log("val/lpips", lpips)
            self.log("val/ssim", ssim)

        if do_visualization:
            depth_pred = torch.cat(depth_pred, dim=0)

            # Construct comparison image.
            vis_prefix = f"{ds_name}/" if ds_name else ""
            context_img = inverse_normalize(batch["context"]["image"][0])
            context = []
            for i in range(context_img.shape[0]):
                context.append(context_img[i])
            comparison = hcat(
                add_label(vcat(*context), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
                add_label(vcat(*depth_pred), "Depth (Prediction)"),
            )

            self.logger.log_image(
                f"{vis_prefix}comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )

            # Visualize GMAE decoder attention maps
            if (
                do_attention_visualization
                and "attention" in visualization_dump
                and visualization_dump["attention"]
            ):
                try:
                    self._visualize_gmae_attention(batch, visualization_dump, gaussian_per_timestamp)
                except Exception as e:
                    print(f"[WARN] attention visualization failed: {e}")

            # Render projections and construct projection image.
            projections = hcat(
                    *render_projections(
                        gaussians,
                        256,
                        extra_label="",
                        low_pass = self.decoder.low_pass_filter,
                    )[0]
                )

            self.logger.log_image(
                f"{vis_prefix}projection",
                [prep_image(add_border(projections))],
                step=self.global_step,
            )

            # Draw cameras.
            cameras = hcat(*render_cameras(batch, 256))
            self.logger.log_image(
                f"{vis_prefix}cameras", [prep_image(add_border(cameras))], step=self.global_step
            )

            if self.encoder_visualizer is not None:
                for k, image in self.encoder_visualizer.visualize(
                    batch["context"], self.global_step
                ).items():
                    self.logger.log_image(k, [prep_image(image)], step=self.global_step)

            # Run video validation step.
            # self.render_video_interpolation(batch)
            # self.render_video_wobble(batch)
            if self.train_cfg.extended_visualization:
                self.render_video_interpolation_exaggerated(batch)

    def compute_lpips_chunked(self, rgb_gt: Tensor, rgb_pred: Tensor) -> Tensor:
        chunk_size = max(1, int(self.train_cfg.validation_lpips_batch_size))
        values = [
            compute_lpips(rgb_gt[start:start + chunk_size], rgb_pred[start:start + chunk_size])
            for start in range(0, rgb_gt.shape[0], chunk_size)
        ]
        return torch.cat(values, dim=0)

    @rank_zero_only
    def on_validation_epoch_end(self):
        if not self._val_metrics:
            return
        all_psnr, all_lpips, all_ssim = [], [], []
        for ds_name, entries in self._val_metrics.items():
            n = len(entries)
            avg_psnr = sum(e["psnr"] for e in entries) / n
            avg_lpips = sum(e["lpips"] for e in entries) / n
            avg_ssim = sum(e["ssim"] for e in entries) / n
            all_psnr.extend(e["psnr"] for e in entries)
            all_lpips.extend(e["lpips"] for e in entries)
            all_ssim.extend(e["ssim"] for e in entries)
            print(f"[VAL] {ds_name}: psnr={avg_psnr:.2f}, lpips={avg_lpips:.4f}, ssim={avg_ssim:.4f} (n={n})")
        # Log total averages across all datasets
        if all_psnr:
            self.log("val/psnr", sum(all_psnr) / len(all_psnr))
            self.log("val/lpips", sum(all_lpips) / len(all_lpips))
            self.log("val/ssim", sum(all_ssim) / len(all_ssim))
            print(f"[VAL] total: psnr={sum(all_psnr)/len(all_psnr):.2f}, "
                  f"lpips={sum(all_lpips)/len(all_lpips):.4f}, "
                  f"ssim={sum(all_ssim)/len(all_ssim):.4f} (n={len(all_psnr)})")
        self._val_metrics.clear()

    def convert_depth_to_normal(self, depth_map, intrinsics):
        B, H, W = depth_map.shape
        intrinsics = intrinsics.clone()
        intrinsics[:,0] = intrinsics[:,0] * W
        intrinsics[:,1] = intrinsics[:,1] * H
        
        pseudo_focal = torch.stack((intrinsics[:,0,0], intrinsics[:,1,1]), dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        pseudo_focal = pseudo_focal.repeat(1,1,H,W)
        
        depth_to_pointmap = depthmap_to_pts3d(depth_map.unsqueeze(dim=-1), pseudo_focal)
        
        pred_normal = utils3d.pt.point_map_to_normal_map(depth_to_pointmap.squeeze(dim=-1))
        
        return pred_normal
        
        
    def get_depth_pseudo_labels(self, images):
        # use vggt to get depth pseudo labels
        if self.train_cfg.reproj_model=='vggt':    
            images = (images * 2) - 1  # normalize to [-1, 1]
            images = images.unsqueeze(dim=1)
            with torch.no_grad():
                output = self.vggt(images)
            depth_pseudo_gt = output['depth'].squeeze(dim=1)
            normal_map = None
        elif self.train_cfg.reproj_model=='moge':
            with torch.no_grad():
                images = rearrange(images, 'b v c h w-> (b v) c h w')
                output = self.vggt.infer(images, resolution_level=5)
            depth_pseudo_gt = output['depth'].clone()
            normal_map = output['normal'].clone()
        return depth_pseudo_gt, normal_map

    @rank_zero_only
    def render_all_timestamps(
        self,
        batch: BatchedExample,
    ) -> None:
        frame_list = []
        timestamps = range(batch['target']['index'].min().item(), batch['target']['index'].max().item() + 1)
        e = batch["context"]["extrinsics"][0, 0]
        i = batch["context"]["intrinsics"][0, 0]
        near = batch["context"]["near"][0, 0]
        far = batch["context"]["far"][0, 0]
        g_cur = self.encoder(batch["context"], self.global_step, target_timestamps = torch.tensor(timestamps, device=self.device))
        for t_idx in timestamps:
            output = self.decoder.forward(
                g_cur[t_idx],
                e[None, None],
                i[None, None],
                near[None, None],
                far[None, None],
                (256, 256),
                depth_mode="depth",
            )
            rgb = output.color[0, 0]
            depth = vis_depth_map(output.depth[0, 0])
            frame_list.append(vcat(rgb, depth))

        video = torch.stack(frame_list)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        visualizations = {
            f"video/all_timestamps": wandb.Video(video[None], fps=5, format="mp4")
        }
        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=5)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                ) 

    @rank_zero_only
    def render_all_timestamps_colored(
        self,
        batch: BatchedExample,
        gaussians: list[Gaussians],
    ) -> None:
        if len(gaussians) < 1:
            return

        frame_list = []
        e = batch["context"]["extrinsics"][0, 0]
        i = batch["context"]["intrinsics"][0, 0]
        near = batch["context"]["near"][0, 0]
        far = batch["context"]["far"][0, 0]

        num_gaussians = gaussians[0].means.shape[1]
        token_colors = self._generate_token_colors(num_gaussians, gaussians[0].means.device)

        for gs in gaussians:
            gs_colored = self._color_gaussians_by_token(gs, token_colors)
            output = self.decoder.forward(
                gs_colored,
                e[None, None],
                i[None, None],
                near[None, None],
                far[None, None],
                (256, 256),
                depth_mode="depth",
            )
            rgb = output.color[0, 0]
            depth = vis_depth_map(output.depth[0, 0])
            frame_list.append(vcat(rgb, depth))

        video = torch.stack(frame_list)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        visualizations = {
            f"video/all_timestamps_colored": wandb.Video(video[None], fps=5, format="mp4")
        }
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=5)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    @rank_zero_only
    def visualize_gaussian_center_tracks(
        self,
        batch: BatchedExample,
        gaussian_per_timestamp: dict,
        max_vis_points: int = 200,
    ) -> None:
        """
        Visualize Gaussian center trajectories from identity camera.
        Produces a side-by-side video with Gaussian center tracks and optional
        CoWTracker tracks on rendered images.
        """
        device = batch["context"]["image"].device
        render_size = 256

        # Identity camera (in relative pose space = first context camera)
        e = torch.eye(4, device=device, dtype=torch.float32)
        intrinsics = batch["context"]["intrinsics"][0, 0]
        near = batch["context"]["near"][0, 0]
        far = batch["context"]["far"][0, 0]

        timestamps = sorted(gaussian_per_timestamp.keys())
        if len(timestamps) < 2:
            return

        total_gaussians = next(iter(gaussian_per_timestamp.values())).means.shape[1]
        num_vis = min(max_vis_points, total_gaussians)
        vis_stride = max(1, total_gaussians // num_vis)
        vis_indices = list(range(0, total_gaussians, vis_stride))[:num_vis]
        token_colors = self._generate_token_colors(num_vis, device)
        token_colors_np = (token_colors * 255).byte().cpu().numpy()

        H, W = render_size, render_size

        # --- First pass: render all frames from identity cam ---
        rendered_frames = []
        all_pts_2d = []
        all_valid = []

        for t_idx in timestamps:
            gs = gaussian_per_timestamp[t_idx]
            output = self.decoder.forward(
                gs, e[None, None], intrinsics[None, None],
                near[None, None], far[None, None],
                (H, W), depth_mode="depth",
            )
            rgb = output.color[0, 0]
            rendered_frames.append(rgb)

            means = gs.means[0][vis_indices]
            pts_2d, valid = project_points_to_image(means, e, intrinsics, (H, W))
            all_pts_2d.append(pts_2d.detach().cpu())
            all_valid.append(valid.detach().cpu())

        # --- Build Gaussian tracks video (left) ---
        trails = [[] for _ in range(num_vis)]
        gs_frame_list = []

        for frame_idx in range(len(timestamps)):
            pts_2d = all_pts_2d[frame_idx]
            valid = all_valid[frame_idx]

            for k in range(num_vis):
                trails[k].append(pts_2d[k] if valid[k] else None)

            img_np = (rendered_frames[frame_idx].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy().copy()

            tail_len = 3
            for k in range(num_vis):
                color_rgb = token_colors_np[k].tolist()
                pts_history = trails[k]
                start = max(1, len(pts_history) - tail_len)
                for j in range(start, len(pts_history)):
                    if pts_history[j - 1] is not None and pts_history[j] is not None:
                        p0 = pts_history[j - 1].int().tolist()
                        p1 = pts_history[j].int().tolist()
                        self._draw_line(img_np, p0[0], p0[1], p1[0], p1[1], color_rgb, thickness=1)

            for k in range(num_vis):
                if valid[k]:
                    x, y = pts_2d[k].int().tolist()
                    if 0 <= y < H and 0 <= x < W:
                        img_np[y, x] = token_colors_np[k].tolist()

            gs_frame_list.append(torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0)

        # --- CoWTracker on rendered video ---
        cow_rendered_list = []
        has_cowtracker = (self.tracking_loss_fn is not None
                         and self.tracking_loss_fn._initialized
                         and hasattr(self.tracking_loss_fn, 'cowtracker'))

        if has_cowtracker:
            cowtracker_size = 224

            # Query points: first-frame Gaussian projected positions
            query_pts = all_pts_2d[0].to(device)
            query_valid = all_valid[0]

            # --- CoWTracker on rendered frames ---
            video_rendered = torch.stack(
                [(rgb.clamp(0, 1) * 255).clamp(0, 255) for rgb in rendered_frames], dim=0
            ).to(device)
            video_rendered = F.interpolate(video_rendered, size=(cowtracker_size, cowtracker_size),
                                           mode='bilinear', align_corners=False)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.tracking_loss_fn.dtype):
                    pred_rendered = self.tracking_loss_fn.cowtracker(video_rendered.unsqueeze(0))

            cow_rendered_list = self._draw_cowtracker_tracks(
                pred_rendered, rendered_frames, query_pts, query_valid,
                num_vis, token_colors_np, H, W
            )

        # --- Compose and log ---
        if len(gs_frame_list) < 2:
            return

        panels = [torch.stack(gs_frame_list)]
        if cow_rendered_list and len(cow_rendered_list) == len(gs_frame_list):
            panels.append(torch.stack(cow_rendered_list))
        combined = torch.cat(panels, dim=-1)  # side-by-side (width)
        combined = (combined.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        visualizations = {"video/gaussian_tracks": wandb.Video(combined[None], fps=3, format="mp4")}

        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=3)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def _draw_cowtracker_tracks(
        self,
        predictions: dict,
        background_frames: list,
        query_pts: torch.Tensor,
        query_valid: torch.Tensor,
        num_gaussians: int,
        token_colors_np,
        H: int, W: int,
    ) -> list:
        """Draw CoWTracker-tracked points on background frames."""
        tracks = predictions["track"][0]
        visibility = predictions["vis"][0]
        confidence = predictions["conf"][0]
        track_H, track_W = tracks.shape[1], tracks.shape[2]

        grid_x = 2 * query_pts[:, 0] / (track_W - 1) - 1
        grid_y = 2 * query_pts[:, 1] / (track_H - 1) - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).clamp(-1, 1)
        grid = grid.unsqueeze(0).unsqueeze(0)

        cow_trails = [[] for _ in range(num_gaussians)]
        frame_list = []

        for frame_idx in range(len(background_frames)):
            track_frame = tracks[frame_idx].permute(2, 0, 1).unsqueeze(0).float()
            vis_frame = visibility[frame_idx].unsqueeze(0).unsqueeze(0).float()
            conf_frame = confidence[frame_idx].unsqueeze(0).unsqueeze(0).float()

            sampled_track = F.grid_sample(track_frame, grid, mode='bilinear',
                                          align_corners=True, padding_mode='border')
            sampled_vis = F.grid_sample(vis_frame, grid, mode='bilinear',
                                        align_corners=True, padding_mode='border')
            sampled_conf = F.grid_sample(conf_frame, grid, mode='bilinear',
                                         align_corners=True, padding_mode='border')

            tracked_pts = sampled_track[0, :, 0, :].T.cpu()
            vis_vals = sampled_vis[0, 0, 0, :].cpu()
            conf_vals = sampled_conf[0, 0, 0, :].cpu()

            tracked_pts[:, 0] *= W / track_W
            tracked_pts[:, 1] *= H / track_H
            quality = vis_vals * conf_vals

            for k in range(num_gaussians):
                if query_valid[k] and quality[k] > 0.1:
                    cow_trails[k].append(tracked_pts[k])
                else:
                    cow_trails[k].append(None)

            img_np = (background_frames[frame_idx].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy().copy()

            tail_len = 3
            for k in range(num_gaussians):
                color_rgb = token_colors_np[k].tolist()
                pts_history = cow_trails[k]
                start = max(1, len(pts_history) - tail_len)
                for j in range(start, len(pts_history)):
                    if pts_history[j - 1] is not None and pts_history[j] is not None:
                        p0 = pts_history[j - 1].int().tolist()
                        p1 = pts_history[j].int().tolist()
                        self._draw_line(img_np, p0[0], p0[1], p1[0], p1[1], color_rgb, thickness=1)

            for k in range(num_gaussians):
                if query_valid[k] and quality[k] > 0.1:
                    x, y = tracked_pts[k].int().tolist()
                    if 0 <= y < H and 0 <= x < W:
                        img_np[y, x] = token_colors_np[k].tolist()

            frame_list.append(torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0)

        return frame_list

    @rank_zero_only
    def visualize_gaussian_flow_map(
        self,
        batch: BatchedExample,
        gaussian_per_timestamp: dict,
    ) -> None:
        """
        Visualize dense optical flow maps derived from Gaussian motion between
        consecutive timestamps.  Produces a side-by-side video:
          Left  – rendered image from the identity camera
          Right – flow map (colour-coded optical flow from Gaussian displacement)
        """
        device = batch["context"]["image"].device
        render_size = 256
        C0 = 0.28209479177387814

        # Identity camera (first context camera in relative pose space)
        e = torch.eye(4, device=device, dtype=torch.float32)
        intrinsics = batch["context"]["intrinsics"][0, 0]
        near = batch["context"]["near"][0, 0]
        far = batch["context"]["far"][0, 0]

        timestamps = sorted(gaussian_per_timestamp.keys())
        if len(timestamps) < 2:
            return

        H, W = render_size, render_size

        # --- First pass: render all frames & project Gaussian means to 2D ---
        rendered_frames = []
        all_pts_2d = []

        for t_idx in timestamps:
            gs = gaussian_per_timestamp[t_idx]
            output = self.decoder.forward(
                gs, e[None, None], intrinsics[None, None],
                near[None, None], far[None, None],
                (H, W), depth_mode="depth",
            )
            rendered_frames.append(output.color[0, 0])

            means = gs.means[0]  # [num_gaussians, 3]
            pts_2d, _ = project_points_to_image(means, e, intrinsics, (H, W))
            all_pts_2d.append(pts_2d.detach())

        # --- Second pass: render per-pair flow maps ---
        frame_list = []
        for i in range(len(timestamps) - 1):
            gs_t = gaussian_per_timestamp[timestamps[i]]

            # Per-Gaussian 2D displacement
            flow_2d = all_pts_2d[i + 1] - all_pts_2d[i]  # [num_gaussians, 2]

            # Normalise to approximately [-1, 1] using 95th percentile
            # (max is dominated by outliers, washing out most flow)
            mag = flow_2d.abs().reshape(-1)
            max_mag = torch.quantile(mag, 0.95).clamp(min=1e-5)
            flow_norm = (flow_2d / max_mag).clamp(-1, 1)
            # Map to [0, 1] for colour-space encoding
            flow_color = (flow_norm + 1) / 2

            # Build flow-encoded Gaussians (only DC component)
            new_harmonics = torch.zeros_like(gs_t.harmonics)
            new_harmonics[0, :, 0, 0] = (flow_color[:, 0] - 0.5) / C0  # u
            new_harmonics[0, :, 1, 0] = (flow_color[:, 1] - 0.5) / C0  # v
            # blue channel: leave at 0 (encodes 0.5 after SH→colour)

            gs_flow = Gaussians(
                means=gs_t.means,
                covariances=gs_t.covariances,
                harmonics=new_harmonics,
                opacities=gs_t.opacities,
                scales=gs_t.scales,
                rotations=gs_t.rotations,
            )

            # Render flow-encoded Gaussians
            output = self.decoder.forward(
                gs_flow, e[None, None], intrinsics[None, None],
                near[None, None], far[None, None],
                (H, W), depth_mode="depth",
            )
            flow_rendered = output.color[0, 0]  # [3, H, W]
            flow_depth = output.depth[0, 0]      # [H, W]

            # Decode back to flow (undo [0,1] → [-1,1])
            flow_u = flow_rendered[0].clamp(0, 1) * 2 - 1
            flow_v = flow_rendered[1].clamp(0, 1) * 2 - 1

            # Mask background (depth == 0 means no Gaussians rendered)
            foreground = flow_depth > 0
            flow_u = (flow_u * foreground.float()).cpu().numpy()
            flow_v = (flow_v * foreground.float()).cpu().numpy()

            # Colour-wheel visualisation
            flow_vis = flow_uv_to_colors(flow_u, flow_v)  # [H, W, 3] uint8
            flow_vis_tensor = torch.from_numpy(flow_vis).permute(2, 0, 1).float() / 255.0

            # Side-by-side: rendered RGB | flow map
            rgb = rendered_frames[i].clamp(0, 1).cpu()
            frame_list.append(hcat(rgb, flow_vis_tensor))

        # Duplicate last frame so video length matches timestamps
        if frame_list:
            frame_list.append(frame_list[-1])

        if len(frame_list) < 2:
            return

        video = torch.stack(frame_list)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        visualizations = {
            "video/gaussian_flow_map": wandb.Video(video[None], fps=3, format="mp4"),
        }
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=3)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    @staticmethod
    def _draw_line(img, x0, y0, x1, y1, color, thickness=1):
        """Draw a line on a numpy image using Bresenham's algorithm."""
        H, W = img.shape[:2]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            for t in range(-thickness, thickness + 1):
                if 0 <= y0 + t < H and 0 <= x0 < W:
                    img[y0 + t, x0] = color
                if 0 <= y0 < H and 0 <= x0 + t < W:
                    img[y0, x0 + t] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    def interpolate_gaussians(
        self,
        g_a: Gaussians,
        g_b: Gaussians,
        alpha: float,
    ) -> Gaussians:
        """
        means: Float[Tensor, "batch gaussian dim"]
        covariances: Float[Tensor, "batch gaussian dim dim"] (from scale and rotation)
        harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
        opacities: Float[Tensor, "batch gaussian"]
        """
        means = (1 - alpha) * g_a.means + alpha * g_b.means
        scales = (1 - alpha) * g_a.scales + alpha * g_b.scales

        # g_a.rotation is quaternion in (w, x, y, z) format
        rotations = slerp_batch(g_a.rotations, g_b.rotations, alpha, quaternion=True)
        # rotations = g_a.rotations  # for simplicity, do not interpolate rotations
        covariances = build_covariance(scales, rotations)
        harmonics = (1 - alpha) * g_a.harmonics + alpha * g_b.harmonics
        opacities = (1 - alpha) * g_a.opacities + alpha * g_b.opacities

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=harmonics,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )

    # def interpolate_extrinsics(
    #     self,
    #     e_a: Tensor,
    #     e_b: Tensor,
    #     alpha: float,
    # ) -> Tensor:
    #     """Interpolate extrinsics between two camera poses."""
    #     r_interp = slerp_batch(e_a[None, :3, :3], e_b[None, :3, :3], alpha, quaternion=False)[0]
    #     t_interp = (1 - alpha) * e_a[:3, 3] + alpha * e_b[:3, 3]
    #     e_interp = torch.eye(4, device=e_a.device)
    #     e_interp[:3, :3] = r_interp
    #     e_interp[:3, 3] = t_interp
    #     return e_interp

    @staticmethod
    def _draw_marker(img, py, px, radius=4, color=(0.0, 1.0, 0.0)):
        """Draw a crosshair marker on image tensor (3, H, W) at (py, px)."""
        _, h, w = img.shape
        py, px = int(round(py)), int(round(px))
        if not (0 <= py < h and 0 <= px < w):
            return img
        img = img.clone()
        color_t = torch.tensor(color, dtype=img.dtype, device=img.device)
        for d in range(-radius, radius + 1):
            # Horizontal bar
            if 0 <= px + d < w:
                img[:, py, px + d] = color_t
            # Vertical bar
            if 0 <= py + d < h:
                img[:, py + d, px] = color_t
        return img

    @staticmethod
    def _project_point_to_view(point_3d, extrinsics, intrinsics, h, w):
        """Project a 3D point onto a view. Extrinsics are C2W, intrinsics are normalized (0-1).
        Returns (pixel_y, pixel_x, is_valid)."""
        # World to camera: w2c = extrinsics.inverse()
        w2c = torch.inverse(extrinsics)  # (4, 4)
        p_homo = torch.cat([point_3d, point_3d.new_ones(1)])  # (4,)
        p_cam = w2c @ p_homo  # (4,)
        p_cam = p_cam[:3]  # (3,)

        if p_cam[2] <= 0:
            return 0, 0, False

        # Normalized 2D: intrinsics @ (p_cam / p_cam[2])
        p_norm = p_cam / p_cam[2]  # (3,)
        p_2d = intrinsics @ p_norm  # (3,) — in 0-1 range

        px = p_2d[0].item() * w
        py = p_2d[1].item() * h
        is_valid = 0 <= px < w and 0 <= py < h
        return py, px, is_valid

    @rank_zero_only
    def _visualize_gmae_attention(self, batch, visualization_dump, gaussian_per_timestamp):
        """Visualize GMAE decoder attention: Gaussian tokens → context patches as spatial heatmaps."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.cm as cm
        import random

        attn_list = visualization_dump['attention']  # list of (B, heads, seq, seq) per layer
        n = visualization_dump['attention_n_patches_per_view']
        num_views = visualization_dump['attention_num_views']
        ctx_index = visualization_dump['attention_context_index']  # (B, v)
        tgt_ts = visualization_dump['attention_target_timestamp']
        num_gaussians = self.encoder.cfg.num_gaussians
        gpt = self.encoder.cfg.gaussians_per_token
        num_ctx_patches = num_views * n
        patch_side = int(n ** 0.5)

        # Random sample one gaussian token
        g_idx = random.randint(0, num_gaussians - 1)

        # Get the 3D positions of ALL sub-gaussians for this token (first batch element)
        gaussians = gaussian_per_timestamp[tgt_ts]
        g_means = [gaussians.means[0, g_idx * gpt + j] for j in range(gpt)]  # list of (3,)

        # Get context images (first batch element, denormalized)
        context_imgs = inverse_normalize(batch["context"]["image"][0])  # (v, 3, H, W)

        # Project all sub-gaussians onto each context view
        # proj_coords_all[vi] = list of (py, px, is_valid) for each sub-gaussian
        h_img, w_img = context_imgs.shape[2], context_imgs.shape[3]
        proj_coords_all = []
        for vi in range(num_views):
            ext = batch["context"]["extrinsics"][0, vi]  # (4, 4) C2W
            intr = batch["context"]["intrinsics"][0, vi]  # (3, 3) normalized
            coords = []
            for g_mean in g_means:
                py, px, valid = self._project_point_to_view(g_mean, ext, intr, h_img, w_img)
                coords.append((py, px, valid))
            proj_coords_all.append(coords)

        # Marker colors: green for gpt==1, cycle through distinct colors for gpt>1
        if gpt == 1:
            marker_colors = [(0.0, 1.0, 0.0)]
        else:
            marker_colors = [
                (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.5, 1.0), (1.0, 1.0, 0.0),
                (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0), (0.5, 1.0, 0.0),
            ]

        # Pick a random single head for single-head visualization
        num_heads = attn_list[0].shape[1]
        rand_head = random.randint(0, num_heads - 1)

        # Build one row per layer for both mean-head and single-head
        rows_mean = []
        rows_single = []
        for layer_idx, attn_layer in enumerate(attn_list):
            for mode, rows_out in [("mean", rows_mean), ("single", rows_single)]:
                if mode == "mean":
                    attn = attn_layer.mean(dim=1)  # (B, seq, seq)
                else:
                    attn = attn_layer[:, rand_head]  # (B, seq, seq)

                # Slice: single gaussian token query → context patch keys
                attn_g2p = attn[0, -num_gaussians + g_idx, :num_ctx_patches]  # (v*n,)
                attn_per_view = attn_g2p.reshape(num_views, patch_side, patch_side)  # (v, PH, PW)

                # Normalize attention across all views (layer-wise)
                layer_min, layer_max = attn_per_view.min(), attn_per_view.max()

                columns = []
                for vi in range(num_views):
                    ctx_img = context_imgs[vi].clamp(0, 1)  # (3, H, W)
                    h_img, w_img = ctx_img.shape[1], ctx_img.shape[2]

                    # Upsample attention map to image resolution
                    attn_map = attn_per_view[vi].unsqueeze(0).unsqueeze(0)
                    attn_map = F.interpolate(attn_map, size=(h_img, w_img), mode='bilinear', align_corners=False)
                    attn_map = attn_map.squeeze(0).squeeze(0)

                    # Normalize to [0, 1] using layer-wise min/max
                    if layer_max > layer_min:
                        attn_map = (attn_map - layer_min) / (layer_max - layer_min)
                    else:
                        attn_map = torch.zeros_like(attn_map)

                    # Apply jet colormap and alpha blend
                    attn_np = attn_map.cpu().numpy()
                    heatmap_rgba = cm.jet(attn_np)
                    heatmap_rgb = torch.tensor(heatmap_rgba[:, :, :3], dtype=torch.float32, device=ctx_img.device)
                    heatmap_rgb = rearrange(heatmap_rgb, 'h w c -> c h w')
                    blended = 0.5 * heatmap_rgb + 0.5 * ctx_img

                    # Draw ALL sub-gaussian markers for this token
                    for j, (py, px, valid) in enumerate(proj_coords_all[vi]):
                        if valid:
                            color = marker_colors[j % len(marker_colors)]
                            radius = 3 if gpt > 1 else 4
                            ctx_img = self._draw_marker(ctx_img, py, px, radius=radius, color=color)
                            blended = self._draw_marker(blended, py, px, radius=radius, color=color)

                    ts_label = f"t={ctx_index[0, vi].item()}"
                    if layer_idx == 0:
                        col = vcat(
                            add_label(ctx_img, ts_label),
                            add_label(blended, f"L{layer_idx}"),
                            gap=4,
                        )
                    else:
                        col = add_label(blended, f"L{layer_idx}")
                    columns.append(col)

                rows_out.append(hcat(*columns, gap=4))

        scene_name = batch["scene"][0] if isinstance(batch["scene"], list) else batch["scene"]

        # Log mean-of-heads visualization
        grid_mean = vcat(*rows_mean, gap=4)
        grid_mean = add_label(grid_mean, f"GMAE Attention mean-head (t={tgt_ts}, token #{g_idx}, {gpt} gs/tok)")
        self.logger.log_image(
            "attention/gmae_decoder",
            [prep_image(add_border(grid_mean))],
            step=self.global_step,
            caption=[f"{scene_name} | t={tgt_ts} | token #{g_idx} | mean-head | {gpt} gs/tok"],
        )

        # Log single-head visualization
        grid_single = vcat(*rows_single, gap=4)
        grid_single = add_label(grid_single, f"GMAE Attention head #{rand_head} (t={tgt_ts}, token #{g_idx}, {gpt} gs/tok)")
        self.logger.log_image(
            "attention/gmae_decoder_single_head",
            [prep_image(add_border(grid_single))],
            step=self.global_step,
            caption=[f"{scene_name} | t={tgt_ts} | token #{g_idx} | head #{rand_head}/{num_heads} | {gpt} gs/tok"],
        )

    @rank_zero_only
    def render_time_interpolation(
        self,
        batch: BatchedExample,
        gaussians: list[Gaussians],
        num_frames: int = 20,
    ) -> None:
        """
        Render time interpolation video given a list of gaussians at different timestamps.
        Interpolates between the gaussians linearly in the parameter space.
        number of timestamps = num_frames + 2
        total frames = number of timestamps - 1 * len(gaussians)
        """
        frame_list = []
        num_timestamps = len(gaussians)
        e = batch["context"]["extrinsics"][0, 0]
        i = batch["context"]["intrinsics"][0, 0]
        near = batch["context"]["near"][0, 0]
        far = batch["context"]["far"][0, 0]
        for t_idx in range(num_timestamps - 1):
            g_a = gaussians[t_idx]
            g_b = gaussians[t_idx + 1]
            for f_idx in range(num_frames):
                alpha = f_idx / num_frames
                g_interp = self.interpolate_gaussians(g_a, g_b, alpha)
                output = self.decoder.forward(
                    g_interp,
                    e[None, None],
                    i[None, None],
                    near[None, None],
                    far[None, None],
                    (256, 256),
                    depth_mode="depth",
                )
                rgb = output.color[0, 0]
                depth = vis_depth_map(output.depth[0, 0])
                frame_list.append(vcat(rgb, depth))

        video = torch.stack(frame_list)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        visualizations = {
            f"video/time_interpolation": wandb.Video(video[None], fps=15, format="mp4")
        }
        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=15)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    @rank_zero_only
    def render_interpolated_timeline(
        self,
        batch: BatchedExample,
        gaussian_per_timestamp: dict,
    ) -> None:
        """
        Render a timeline video comparing GT vs rendered (interpolated) images.
        Context timestamps: rendered from encoder-predicted gaussians.
        Target timestamps: rendered from gaussians interpolated between nearest two context timestamps.
        Each frame is rendered at its corresponding camera position for GT comparison.
        Output: side-by-side video (GT top, rendered bottom) sorted by timestamp.
        """
        # Collect context timestamps (unique, sorted)
        ctx_indices = batch["context"]["index"][0].tolist()
        ctx_ts_sorted = sorted(set(ctx_indices))

        # Build list of (timestamp, extrinsics, intrinsics, near, far, gt_image, is_context)
        frames = []

        # Context views
        for v_idx in range(batch["context"]["index"].shape[1]):
            t = batch["context"]["index"][0, v_idx].item()
            frames.append((
                t,
                batch["context"]["extrinsics"][0, v_idx],
                batch["context"]["intrinsics"][0, v_idx],
                batch["context"]["near"][0, v_idx],
                batch["context"]["far"][0, v_idx],
                (batch["context"]["image"][0, v_idx] + 1) / 2,  # [-1,1] -> [0,1]
                True,  # is_context
            ))

        # Target views
        for v_idx in range(batch["target"]["index"].shape[1]):
            t = batch["target"]["index"][0, v_idx].item()
            frames.append((
                t,
                batch["target"]["extrinsics"][0, v_idx],
                batch["target"]["intrinsics"][0, v_idx],
                batch["target"]["near"][0, v_idx],
                batch["target"]["far"][0, v_idx],
                batch["target"]["image"][0, v_idx],  # already [0,1]
                False,  # is_context
            ))

        # Sort by timestamp
        frames.sort(key=lambda x: x[0])

        gt_list = []
        rendered_list = []

        for t, e, i, near, far, gt_img, is_context in frames:
            if is_context:
                # Use encoder-predicted gaussians directly
                gs = gaussian_per_timestamp[t]
            else:
                # Interpolate between nearest two context timestamps
                lower = [ct for ct in ctx_ts_sorted if ct <= t]
                upper = [ct for ct in ctx_ts_sorted if ct >= t]

                if not lower:
                    gs = gaussian_per_timestamp[ctx_ts_sorted[0]]
                elif not upper:
                    gs = gaussian_per_timestamp[ctx_ts_sorted[-1]]
                elif lower[-1] == upper[0]:
                    gs = gaussian_per_timestamp[t]
                else:
                    t_a, t_b = lower[-1], upper[0]
                    alpha = (t - t_a) / (t_b - t_a)
                    gs = self.interpolate_gaussians(
                        gaussian_per_timestamp[t_a],
                        gaussian_per_timestamp[t_b],
                        alpha,
                    )

            output = self.decoder.forward(
                gs,
                e[None, None],
                i[None, None],
                near[None, None],
                far[None, None],
                (256, 256),
                depth_mode="depth",
            )
            rgb = output.color[0, 0]

            # Resize GT to 256x256 to match rendered
            gt_resized = F.interpolate(
                gt_img[None], size=(256, 256), mode="bilinear", align_corners=False
            )[0]

            gt_list.append(gt_resized)
            rendered_list.append(rgb)

        # Stack: GT on top, rendered on bottom
        combined = []
        for gt_f, rend_f in zip(gt_list, rendered_list):
            combined.append(vcat(gt_f, rend_f))

        video = torch.stack(combined)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        visualizations = {
            "video/interpolated_timeline": wandb.Video(video[None], fps=5, format="mp4"),
        }
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=5)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def _generate_token_colors(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        """
        Generate distinct colors for each token index using HSV colormap.

        Args:
            num_tokens: Number of tokens/Gaussians
            device: Target device

        Returns:
            colors: [num_tokens, 3] RGB colors in range [0, 1]
        """
        # Use HSV colormap for maximally distinct colors
        hues = torch.linspace(0, 1, num_tokens + 1, device=device)[:-1]  # Exclude 1 to avoid duplicate with 0

        # Convert HSV to RGB (S=1, V=1 for vivid colors)
        colors = torch.zeros(num_tokens, 3, device=device)

        for i, h in enumerate(hues):
            # HSV to RGB conversion
            h_i = int(h * 6)
            f = h * 6 - h_i

            if h_i == 0:
                colors[i] = torch.tensor([1, f, 0], device=device)
            elif h_i == 1:
                colors[i] = torch.tensor([1 - f, 1, 0], device=device)
            elif h_i == 2:
                colors[i] = torch.tensor([0, 1, f], device=device)
            elif h_i == 3:
                colors[i] = torch.tensor([0, 1 - f, 1], device=device)
            elif h_i == 4:
                colors[i] = torch.tensor([f, 0, 1], device=device)
            else:
                colors[i] = torch.tensor([1, 0, 1 - f], device=device)

        return colors

    def _color_gaussians_by_token(self, gaussians: Gaussians, token_colors: torch.Tensor) -> Gaussians:
        """
        Color Gaussians based on their token index.

        Args:
            gaussians: Original Gaussians
            token_colors: [num_tokens, 3] RGB colors

        Returns:
            New Gaussians with modified harmonics for coloring
        """
        # Spherical harmonics DC coefficient
        C0 = 0.28209479177387814

        # Get number of Gaussians
        batch_size, num_gaussians, _, d_sh = gaussians.harmonics.shape

        # Create new harmonics with token colors
        new_harmonics = gaussians.harmonics.clone()

        # Set DC component (index 0) to the token color
        # Formula: color = SH_coeff * C0 + 0.5, so SH_coeff = (color - 0.5) / C0
        for g_idx in range(num_gaussians):
            color = token_colors[g_idx % len(token_colors)]  # Handle case where num_gaussians > num_colors
            new_harmonics[:, g_idx, :, 0] = (color - 0.5) / C0

        return Gaussians(
            means=gaussians.means,
            covariances=gaussians.covariances,
            harmonics=new_harmonics,
            opacities=gaussians.opacities,
            scales=gaussians.scales,
            rotations=gaussians.rotations,
        )

    @rank_zero_only
    def render_time_interpolation_colored(
        self,
        batch: BatchedExample,
        gaussians: list[Gaussians],
        num_frames: int = 20,
    ) -> None:
        """
        Render time interpolation video with Gaussians colored by their token index.
        Each Gaussian that originated from the same learnable query token will have
        the same color, making it easy to track their motion across time.

        Args:
            batch: Input batch with camera parameters
            gaussians: List of Gaussians, one per timestamp
            num_frames: Number of interpolation frames between each pair of timestamps
        """
        if len(gaussians) < 2:
            return

        frame_list = []
        num_timestamps = len(gaussians)
        e = batch["context"]["extrinsics"][0, 0]
        i = batch["context"]["intrinsics"][0, 0]
        near = batch["context"]["near"][0, 0]
        far = batch["context"]["far"][0, 0]

        # Generate colors for each token
        num_gaussians = gaussians[0].means.shape[1]
        token_colors = self._generate_token_colors(num_gaussians, gaussians[0].means.device)

        for t_idx in range(num_timestamps - 1):
            g_a = gaussians[t_idx]
            g_b = gaussians[t_idx + 1]

            # Color both sets of Gaussians
            g_a_colored = self._color_gaussians_by_token(g_a, token_colors)
            g_b_colored = self._color_gaussians_by_token(g_b, token_colors)

            for f_idx in range(num_frames):
                alpha = f_idx / num_frames
                g_interp = self.interpolate_gaussians(g_a_colored, g_b_colored, alpha)

                output = self.decoder.forward(
                    g_interp,
                    e[None, None],
                    i[None, None],
                    near[None, None],
                    far[None, None],
                    (256, 256),
                    depth_mode="depth",
                )
                rgb = output.color[0, 0]
                depth = vis_depth_map(output.depth[0, 0])
                frame_list.append(vcat(rgb, depth))

        video = torch.stack(frame_list)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        visualizations = {
            f"video/time_interpolation_colored": wandb.Video(video[None], fps=15, format="mp4")
        }

        # Log to wandb or local
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=15)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians = self.encoder(batch["context"], self.global_step)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = ImageSequenceClip(list(tensor), fps=30)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian_param_head" in name or "intrinsic_encoder" in name or 'gmae' in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
