"""
Point Tracking Consistency Loss (Multi-Timestamp, Per-Camera-Trajectory)

This loss ensures that learnable Gaussian tokens produce means that are consistent
with point tracking predictions across different timestamps within the same camera trajectory.

Given:
- Multiple sets of Gaussians, one per timestamp
- Context and target images organized by camera trajectory and timestamp
- A point tracking model (CoWTracker)

The loss:
1. For each camera trajectory:
   - Select one timestamp as the query view
   - Project Gaussians (from that timestamp) onto the query view to get 2D query points
   - Use CoWTracker to track these points to other timestamps (same camera, different time)
   - Project Gaussians from OTHER timestamps to their respective views
   - Penalize discrepancy between tracked points and projected points

This enforces that the same learnable token should produce Gaussian means that
project consistently across time when viewed from the same camera.

Visualization functions are also provided to debug and verify the tracking predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys
import os

# Add cowtracker to path


def project_points_to_image(
    points_3d: torch.Tensor,  # [B, N, 3] or [N, 3] world coordinates
    extrinsics: torch.Tensor,  # [B, 4, 4] or [4, 4]
    intrinsics: torch.Tensor,  # [B, 3, 3] or [3, 3]
    image_size: tuple[int, int],  # (H, W)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: [B, N, 3] or [N, 3] 3D points in world coordinates
        extrinsics: [B, 4, 4] or [4, 4] camera extrinsics (world to camera)
        intrinsics: [B, 3, 3] or [3, 3] camera intrinsics (normalized, 0-1 range)
        image_size: (H, W) image dimensions

    Returns:
        points_2d: [B, N, 2] or [N, 2] 2D points in pixel coordinates (x, y)
        valid_mask: [B, N] or [N] boolean mask for points in front of camera and within bounds
    """
    # Handle unbatched inputs
    squeeze_batch = False
    if points_3d.dim() == 2:
        points_3d = points_3d.unsqueeze(0)
        extrinsics = extrinsics.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch = True

    B, N, _ = points_3d.shape
    H, W = image_size

    # Convert to homogeneous coordinates [B, N, 4]
    ones = torch.ones_like(points_3d[..., :1])
    points_homo = torch.cat([points_3d, ones], dim=-1)

    # Transform to camera coordinates: [B, 4, 4] @ [B, N, 4].T -> [B, N, 4]
    extrinsics_inv = torch.linalg.inv(extrinsics)
    points_cam = torch.einsum('bij,bnj->bni', extrinsics_inv, points_homo)[..., :3]  # [B, N, 3]

    # Check if points are in front of camera (z > 0)
    valid_mask = points_cam[..., 2] > 0  # [B, N]

    # Perspective division
    eps = 1e-6
    z = points_cam[..., 2:3].clamp(min=eps)  # [B, N, 1]
    points_normalized = points_cam[..., :2] / z  # [B, N, 2] - (x/z, y/z)

    # Apply intrinsics (convert normalized intrinsics to pixel coordinates)
    # intrinsics are normalized (0-1), so we need to scale
    fx = intrinsics[:, 0, 0:1] * W  # [B, 1]
    fy = intrinsics[:, 1, 1:2] * H  # [B, 1]
    cx = intrinsics[:, 0, 2:3] * W  # [B, 1]
    cy = intrinsics[:, 1, 2:3] * H  # [B, 1]

    # [B, N, 2]
    points_2d_x = points_normalized[..., 0] * fx + cx  # [B, N]
    points_2d_y = points_normalized[..., 1] * fy + cy  # [B, N]
    points_2d = torch.stack([points_2d_x, points_2d_y], dim=-1)  # [B, N, 2]

    # Additional validity check: points within image bounds
    valid_mask = valid_mask & (points_2d[..., 0] >= 0) & (points_2d[..., 0] < W)
    valid_mask = valid_mask & (points_2d[..., 1] >= 0) & (points_2d[..., 1] < H)

    if squeeze_batch:
        points_2d = points_2d.squeeze(0)
        valid_mask = valid_mask.squeeze(0)

    return points_2d, valid_mask


class TrackingConsistencyLoss(nn.Module):
    """
    Loss that enforces consistency between Gaussian projections and point tracking predictions
    across different timestamps within the same camera trajectory.
    """

    def __init__(
        self,
        cowtracker_path: str | None = None,
        num_query_points: int = 256,
        device: str = "cuda",
        dtype=torch.float16,
    ):
        super().__init__()
        self.num_query_points = num_query_points
        self.cowtracker = None
        self.cowtracker_path = cowtracker_path
        self.device_str = device
        self.dtype = dtype
        self._initialized = False
        self._lazy_init(torch.device(self.device_str))

    def _lazy_init(self, device):
        """Lazily initialize CoWTracker on first use."""
        if self._initialized:
            return

        from submodules.cowtracker.cowtracker import CoWTracker

        print("Initializing CoWTracker for tracking consistency loss...")
        self.cowtracker = CoWTracker.from_checkpoint(
            checkpoint_path=self.cowtracker_path,
            device=device,
            dtype=self.dtype,
        )
        self.cowtracker.eval()
        for p in self.cowtracker.parameters():
            p.requires_grad = False

        self._initialized = True
        print("CoWTracker initialized successfully!")

    def forward(
        self,
        gaussians_per_timestamp: dict,  # {timestamp: Gaussians} where Gaussians.means is [B, num_gaussians, 3]
        context_images: torch.Tensor,  # [B, num_context, 3, H, W]
        context_extrinsics: torch.Tensor,  # [B, num_context, 4, 4]
        context_intrinsics: torch.Tensor,  # [B, num_context, 3, 3]
        context_timestamps: torch.Tensor,  # [B, num_context] timestamp index for each context view
        context_cameras: torch.Tensor,  # [B, num_context] camera ID for each context view
        target_images: torch.Tensor,  # [B, num_target, 3, H, W]
        target_extrinsics: torch.Tensor,  # [B, num_target, 4, 4]
        target_intrinsics: torch.Tensor,  # [B, num_target, 3, 3]
        target_timestamps: torch.Tensor,  # [B, num_target] timestamp index for each target view
        target_cameras: torch.Tensor,  # [B, num_target] camera ID for each target view
    ) -> torch.Tensor:
        """
        Compute tracking consistency loss across timestamps within each camera trajectory.

        For each camera trajectory:
        1. Find all views (context + target) from this camera at different timestamps
        2. Pick one timestamp as query, project its Gaussians to get 2D query points
        3. Track these points to other timestamps using CoWTracker
        4. Compare tracked positions with actual projections of Gaussians from those timestamps

        Returns:
            loss: scalar tensor
        """
        device = context_images.device
        self._lazy_init(device)

        B, num_context, _, H, W = context_images.shape
        _, num_target, _, _, _ = target_images.shape

        total_loss = 0.0
        num_valid_pairs = 0

        for b in range(B):
            # Get unique camera IDs in this batch
            ctx_cameras_b = context_cameras[b] if context_cameras is not None else torch.arange(num_context, device=device)
            tgt_cameras_b = target_cameras[b] if target_cameras is not None else torch.arange(num_target, device=device)

            all_camera_ids = torch.unique(torch.cat([ctx_cameras_b, tgt_cameras_b]))

            for cam_id in all_camera_ids.tolist():
                # Find all context views from this camera
                ctx_mask = (ctx_cameras_b == cam_id) if context_cameras is not None else torch.zeros(num_context, dtype=torch.bool, device=device)
                tgt_mask = (tgt_cameras_b == cam_id) if target_cameras is not None else torch.zeros(num_target, dtype=torch.bool, device=device)

                # Collect (image, extrinsics, intrinsics, timestamp) for this camera
                cam_data = []

                for i in range(num_context):
                    if ctx_mask[i] if context_cameras is not None else False:
                        t = context_timestamps[b, i].item()
                        cam_data.append({
                            'image': context_images[b, i],  # [3, H, W]
                            'extrinsics': context_extrinsics[b, i],  # [4, 4]
                            'intrinsics': context_intrinsics[b, i],  # [3, 3]
                            'timestamp': t,
                            'is_context': True,
                        })

                for i in range(num_target):
                    if tgt_mask[i] if target_cameras is not None else False:
                        t = target_timestamps[b, i].item()
                        cam_data.append({
                            'image': target_images[b, i],  # [3, H, W]
                            'extrinsics': target_extrinsics[b, i],  # [4, 4]
                            'intrinsics': target_intrinsics[b, i],  # [3, 3]
                            'timestamp': t,
                            'is_context': False,
                        })

                # Need at least 2 views from same camera at different timestamps
                if len(cam_data) < 2:
                    continue

                # Sort by timestamp
                cam_data = sorted(cam_data, key=lambda x: x['timestamp'])
                timestamps_in_cam = [d['timestamp'] for d in cam_data]
                unique_timestamps = list(set(timestamps_in_cam))

                if len(unique_timestamps) < 2:
                    continue

                # Use first timestamp as query
                query_idx = 0
                query_data = cam_data[query_idx]
                query_t = query_data['timestamp']

                # Check if we have Gaussians for the query timestamp
                if query_t not in gaussians_per_timestamp:
                    continue

                query_means = gaussians_per_timestamp[query_t].means[b]  # [num_gaussians, 3]

                # Project Gaussians to query view
                query_2d, query_valid = project_points_to_image(
                    query_means,
                    query_data['extrinsics'],
                    query_data['intrinsics'],
                    (H, W)
                )  # [num_gaussians, 2], [num_gaussians]

                # Sample valid points
                valid_indices = query_valid.nonzero(as_tuple=True)[0]
                if len(valid_indices) < 10:
                    continue

                # num_samples = min(self.num_query_points, len(valid_indices)
                # perm = torch.randperm(len(valid_indices), device=device)[:num_samples]
                sampled_indices = valid_indices

                sampled_query_2d = query_2d[sampled_indices]  # [num_samples, 2]

                # Prepare video for CoWTracker: all views from this camera sorted by timestamp
                video_frames = []
                for d in cam_data:
                    img = d['image']  # [3, H, W]
                    # Convert from [-1, 1] or [0, 1] to [0, 255]
                    if img.min() < 0:
                        img = (img + 1) / 2
                    img = (img * 255).clamp(0, 255)
                    video_frames.append(img)

                video = torch.stack(video_frames, dim=0)  # [T, 3, H, W]

                # Resize to CoWTracker compatible size (must be divisible by patch_size=14)
                # Using 336x336 which is 24*14 = 336
                cowtracker_size = 224  # Divisible by 14
                video = F.interpolate(video, size=(cowtracker_size, cowtracker_size), mode='bilinear', align_corners=False)
                sampled_query_2d_scaled = sampled_query_2d.clone()

                # Run CoWTracker
                video_input = video.unsqueeze(0)  # [1, T, 3, cowtracker_size, cowtracker_size]
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=self.dtype):
                        predictions = self.cowtracker(video_input)

                tracks = predictions["track"][0]  # [T, H_out, W_out, 2] - pixel coordinates
                visibility = predictions["vis"][0]  # [T, H_out, W_out]
                confidence = predictions["conf"][0]  # [T, H_out, W_out]

                # Get actual output dimensions from CoWTracker
                track_H, track_W = tracks.shape[1], tracks.shape[2]

                # For each target frame (other timestamps), compute loss
                for frame_idx, target_data in enumerate(cam_data):
                    if frame_idx == query_idx:
                        continue  # Skip query frame

                    target_t = target_data['timestamp']

                    # Check if we have Gaussians for this timestamp
                    if target_t not in gaussians_per_timestamp:
                        continue

                    target_means = gaussians_per_timestamp[target_t].means[b]  # [num_gaussians, 3]
                    sampled_target_means = target_means[sampled_indices]  # [num_samples, 3]

                    # Sample tracks at query point locations using bilinear interpolation
                    # Query points are in cowtracker_size space, tracks output is also in that space

                    # Normalize query points to [-1, 1] for grid_sample
                    # grid_sample expects (x, y) in [-1, 1] where -1 is left/top and 1 is right/bottom
                    grid_x = 2 * sampled_query_2d_scaled[:, 0] / (track_W - 1) - 1  # [num_samples]
                    grid_y = 2 * sampled_query_2d_scaled[:, 1] / (track_H - 1) - 1  # [num_samples]
                    grid = torch.stack([grid_x, grid_y], dim=-1)  # [num_samples, 2]
                    grid = grid.clamp(-1, 1)
                    grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, num_samples, 2]

                    # Sample from track, visibility, and confidence maps
                    # tracks[frame_idx] is [H, W, 2], need to reshape for grid_sample
                    track_frame = tracks[frame_idx].permute(2, 0, 1).unsqueeze(0).float()  # [1, 2, H, W]
                    vis_frame = visibility[frame_idx].unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
                    conf_frame = confidence[frame_idx].unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

                    sampled_track = F.grid_sample(
                        track_frame, grid, mode='bilinear', align_corners=True, padding_mode='border'
                    )  # [1, 2, 1, num_samples]
                    sampled_vis = F.grid_sample(
                        vis_frame, grid, mode='bilinear', align_corners=True, padding_mode='border'
                    )  # [1, 1, 1, num_samples]
                    sampled_conf = F.grid_sample(
                        conf_frame, grid, mode='bilinear', align_corners=True, padding_mode='border'
                    )  # [1, 1, 1, num_samples]

                    tracked_pts = sampled_track[0, :, 0, :].T  # [num_samples, 2]
                    vis_t = sampled_vis[0, 0, 0, :]  # [num_samples]
                    conf_t = sampled_conf[0, 0, 0, :]  # [num_samples]

                    # Project target Gaussians to target view
                    projected_2d, proj_valid = project_points_to_image(
                        sampled_target_means,
                        target_data['extrinsics'],
                        target_data['intrinsics'],
                        (H, W)
                    )  # [num_samples, 2], [num_samples]

                    # Compute quality weight
                    quality = vis_t * conf_t

                    # Only compute loss for valid projections and high-quality tracks
                    valid = (proj_valid & (quality > 0.1)).detach()

                    if valid.sum() > 0:
                        # L2 loss between projected and tracked points
                        # diff = projected_2d[valid] - tracked_pts[valid]
                        
                        loss = F.huber_loss(projected_2d[valid], tracked_pts[valid].detach(), delta=0.01*max(H, W))

                        # Weighted MSE loss (normalized by image size)
                        # loss = (weights * (diff ** 2).sum(dim=-1)).sum() / (weights.sum() + 1e-6)
                        loss = loss / max(H, W)  # Normalize by image size

                        total_loss = total_loss + loss
                        num_valid_pairs += 1

        if num_valid_pairs > 0:
            return total_loss / num_valid_pairs
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


def visualize_tracking_consistency(
    gaussians_per_timestamp: dict,  # {timestamp: means tensor [B, num_gaussians, 3]}
    context_images: torch.Tensor,  # [B, num_context, 3, H, W]
    context_extrinsics: torch.Tensor,  # [B, num_context, 4, 4]
    context_intrinsics: torch.Tensor,  # [B, num_context, 3, 3]
    context_timestamps: torch.Tensor,  # [B, num_context]
    context_cameras: torch.Tensor,  # [B, num_context]
    target_images: torch.Tensor,  # [B, num_target, 3, H, W]
    target_extrinsics: torch.Tensor,  # [B, num_target, 4, 4]
    target_intrinsics: torch.Tensor,  # [B, num_target, 3, 3]
    target_timestamps: torch.Tensor,  # [B, num_target]
    target_cameras: torch.Tensor,  # [B, num_target]
    cowtracker_model,
    output_dir: str,
    num_points_to_show: int = 50,
    batch_idx: int = 0,
):
    """
    Visualize tracking consistency by showing:
    1. Query points on the first frame
    2. Tracked points vs projected points on subsequent frames
    3. Error vectors between tracked and projected positions

    Args:
        gaussians_per_timestamp: Dict mapping timestamp to Gaussian means
        context_images: Context view images
        context_extrinsics: Context camera extrinsics
        context_intrinsics: Context camera intrinsics
        context_timestamps: Timestamp for each context view
        context_cameras: Camera ID for each context view
        target_images: Target view images
        target_extrinsics: Target camera extrinsics
        target_intrinsics: Target camera intrinsics
        target_timestamps: Timestamp for each target view
        target_cameras: Camera ID for each target view
        cowtracker_model: Initialized CoWTracker model
        output_dir: Directory to save visualizations
        num_points_to_show: Number of points to visualize
        batch_idx: Which batch element to visualize
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(output_dir, exist_ok=True)

    device = context_images.device
    B, num_context, _, H, W = context_images.shape
    _, num_target, _, _, _ = target_images.shape

    b = batch_idx

    # Get camera IDs
    ctx_cameras_b = context_cameras[b] if context_cameras is not None else torch.arange(num_context, device=device)
    tgt_cameras_b = target_cameras[b] if target_cameras is not None else torch.arange(num_target, device=device)
    all_camera_ids = torch.unique(torch.cat([ctx_cameras_b, tgt_cameras_b]))

    for cam_id in all_camera_ids.tolist():
        # Find all views from this camera
        ctx_mask = (ctx_cameras_b == cam_id) if context_cameras is not None else torch.zeros(num_context, dtype=torch.bool, device=device)
        tgt_mask = (tgt_cameras_b == cam_id) if target_cameras is not None else torch.zeros(num_target, dtype=torch.bool, device=device)

        # Collect data for this camera
        cam_data = []
        for i in range(num_context):
            if ctx_mask[i] if context_cameras is not None else False:
                t = context_timestamps[b, i].item()
                cam_data.append({
                    'image': context_images[b, i],
                    'extrinsics': context_extrinsics[b, i],
                    'intrinsics': context_intrinsics[b, i],
                    'timestamp': t,
                })

        for i in range(num_target):
            if tgt_mask[i] if target_cameras is not None else False:
                t = target_timestamps[b, i].item()
                cam_data.append({
                    'image': target_images[b, i],
                    'extrinsics': target_extrinsics[b, i],
                    'intrinsics': target_intrinsics[b, i],
                    'timestamp': t,
                })

        if len(cam_data) < 2:
            continue

        # Sort by timestamp
        cam_data = sorted(cam_data, key=lambda x: x['timestamp'])
        unique_timestamps = list(set([d['timestamp'] for d in cam_data]))

        if len(unique_timestamps) < 2:
            continue

        # Use first timestamp as query
        query_data = cam_data[0]
        query_t = query_data['timestamp']

        if query_t not in gaussians_per_timestamp:
            continue

        query_means = gaussians_per_timestamp[query_t].means[b]  # [num_gaussians, 3]

        # Project to query view
        query_2d, query_valid = project_points_to_image(
            query_means, query_data['extrinsics'], query_data['intrinsics'], (H, W)
        )

        # Sample valid points
        valid_indices = query_valid.nonzero(as_tuple=True)[0]
        if len(valid_indices) < 10:
            continue

        num_samples = min(num_points_to_show, len(valid_indices))
        perm = torch.randperm(len(valid_indices), device=device)[:num_samples]
        sampled_indices = valid_indices[perm]
        sampled_query_2d = query_2d[sampled_indices].detach()

        # Prepare video for CoWTracker
        video_frames = []
        for d in cam_data:
            img = d['image'].clone()
            if img.min() < 0:
                img = (img + 1) / 2
            img = (img * 255).clamp(0, 255)
            video_frames.append(img)

        video = torch.stack(video_frames, dim=0)
        orig_H, orig_W = H, W
        cowtracker_size = 336
        video = F.interpolate(video, size=(cowtracker_size, cowtracker_size), mode='bilinear', align_corners=False)

        # Scale query points
        scale_x = cowtracker_size / orig_W
        scale_y = cowtracker_size / orig_H
        sampled_query_2d_scaled = sampled_query_2d.clone()
        sampled_query_2d_scaled[:, 0] *= scale_x
        sampled_query_2d_scaled[:, 1] *= scale_y

        # Run CoWTracker
        video_input = video.unsqueeze(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                predictions = cowtracker_model(video_input)

        tracks = predictions["track"][0]  # [T, H_out, W_out, 2]
        visibility = predictions["vis"][0]
        confidence = predictions["conf"][0]
        track_H, track_W = tracks.shape[1], tracks.shape[2]

        # Generate distinct colors for each point
        colors = plt.cm.rainbow(np.linspace(0, 1, num_samples))

        # Create figure with subplots for each frame
        num_frames = len(cam_data)
        fig, axes = plt.subplots(2, num_frames, figsize=(5 * num_frames, 10))
        if num_frames == 1:
            axes = axes.reshape(2, 1)

        for frame_idx, frame_data in enumerate(cam_data):
            frame_t = frame_data['timestamp']
            img = frame_data['image'].clone()
            if img.min() < 0:
                img = (img + 1) / 2
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)

            # Top row: Show tracked points
            axes[0, frame_idx].imshow(img_np)
            axes[0, frame_idx].set_title(f"Frame {frame_idx} (t={frame_t})\nTracked Points")

            # Bottom row: Show projected points and errors
            axes[1, frame_idx].imshow(img_np)
            axes[1, frame_idx].set_title(f"Frame {frame_idx} (t={frame_t})\nProjected vs Tracked")

            if frame_idx == 0:
                # Query frame - just show query points
                for i, (pt, color) in enumerate(zip(sampled_query_2d.cpu().numpy(), colors)):
                    axes[0, frame_idx].scatter(pt[0], pt[1], c=[color], s=30, marker='o')
                    axes[1, frame_idx].scatter(pt[0], pt[1], c=[color], s=30, marker='o')
                axes[0, frame_idx].set_title(f"Frame {frame_idx} (t={frame_t})\nQuery Points")
            else:
                # Sample tracked points
                grid_x = 2 * sampled_query_2d_scaled[:, 0] / (track_W - 1) - 1
                grid_y = 2 * sampled_query_2d_scaled[:, 1] / (track_H - 1) - 1
                grid = torch.stack([grid_x, grid_y], dim=-1).clamp(-1, 1)
                grid = grid.unsqueeze(0).unsqueeze(0)

                track_frame = tracks[frame_idx].permute(2, 0, 1).unsqueeze(0).float()
                vis_frame = visibility[frame_idx].unsqueeze(0).unsqueeze(0).float()
                conf_frame = confidence[frame_idx].unsqueeze(0).unsqueeze(0).float()

                sampled_track = F.grid_sample(track_frame, grid, mode='bilinear', align_corners=True, padding_mode='border')
                sampled_vis = F.grid_sample(vis_frame, grid, mode='bilinear', align_corners=True, padding_mode='border')
                sampled_conf = F.grid_sample(conf_frame, grid, mode='bilinear', align_corners=True, padding_mode='border')

                tracked_pts = sampled_track[0, :, 0, :].T.cpu().numpy()  # [num_samples, 2]
                vis_vals = sampled_vis[0, 0, 0, :].cpu().numpy()
                conf_vals = sampled_conf[0, 0, 0, :].cpu().numpy()

                # Scale back to original size
                tracked_pts[:, 0] *= orig_W / track_W
                tracked_pts[:, 1] *= orig_H / track_H

                # Get projected points for this timestamp
                if frame_t in gaussians_per_timestamp:
                    target_means = gaussians_per_timestamp[frame_t].means[b]
                    sampled_target_means = target_means[sampled_indices]
                    projected_2d, proj_valid = project_points_to_image(
                        sampled_target_means, frame_data['extrinsics'], frame_data['intrinsics'], (orig_H, orig_W)
                    )
                    projected_pts = projected_2d.detach().cpu().numpy()
                    proj_valid_np = proj_valid.detach().cpu().numpy()
                else:
                    projected_pts = None
                    proj_valid_np = None

                # Plot tracked points (top row)
                for i, (pt, color, vis, conf) in enumerate(zip(tracked_pts, colors, vis_vals, conf_vals)):
                    alpha = vis * conf
                    if alpha > 0.1:
                        axes[0, frame_idx].scatter(pt[0], pt[1], c=[color], s=30, marker='o', alpha=float(alpha))

                # Plot projected vs tracked (bottom row)
                if projected_pts is not None:
                    for i in range(num_samples):
                        color = colors[i]
                        vis, conf = vis_vals[i], conf_vals[i]
                        alpha = vis * conf

                        if proj_valid_np[i] and alpha > 0.1:
                            proj_pt = projected_pts[i]
                            track_pt = tracked_pts[i]

                            # Draw projected point (circle)
                            axes[1, frame_idx].scatter(proj_pt[0], proj_pt[1], c=[color], s=50, marker='o', label='Projected' if i == 0 else '')

                            # Draw tracked point (x)
                            axes[1, frame_idx].scatter(track_pt[0], track_pt[1], c=[color], s=50, marker='x', alpha=float(alpha))

                            # Draw error vector
                            axes[1, frame_idx].arrow(
                                proj_pt[0], proj_pt[1],
                                track_pt[0] - proj_pt[0], track_pt[1] - proj_pt[1],
                                head_width=3, head_length=2, fc=color, ec=color, alpha=0.5
                            )

            axes[0, frame_idx].axis('off')
            axes[1, frame_idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"tracking_vis_cam{int(cam_id)}_batch{b}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved tracking visualization to {save_path}")

        # Also create a summary plot showing error statistics
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

        all_errors = []
        all_vis_conf = []

        for frame_idx, frame_data in enumerate(cam_data[1:], start=1):
            frame_t = frame_data['timestamp']
            if frame_t not in gaussians_per_timestamp:
                continue

            # Recompute for statistics
            grid_x = 2 * sampled_query_2d_scaled[:, 0] / (track_W - 1) - 1
            grid_y = 2 * sampled_query_2d_scaled[:, 1] / (track_H - 1) - 1
            grid = torch.stack([grid_x, grid_y], dim=-1).clamp(-1, 1).unsqueeze(0).unsqueeze(0)

            track_frame = tracks[frame_idx].permute(2, 0, 1).unsqueeze(0).float()
            vis_frame = visibility[frame_idx].unsqueeze(0).unsqueeze(0).float()
            conf_frame = confidence[frame_idx].unsqueeze(0).unsqueeze(0).float()

            sampled_track = F.grid_sample(track_frame, grid, mode='bilinear', align_corners=True, padding_mode='border')
            sampled_vis = F.grid_sample(vis_frame, grid, mode='bilinear', align_corners=True, padding_mode='border')
            sampled_conf = F.grid_sample(conf_frame, grid, mode='bilinear', align_corners=True, padding_mode='border')

            tracked_pts = sampled_track[0, :, 0, :].T
            tracked_pts[:, 0] *= orig_W / track_W
            tracked_pts[:, 1] *= orig_H / track_H

            target_means = gaussians_per_timestamp[frame_t].means[b]
            sampled_target_means = target_means[sampled_indices]
            projected_2d, proj_valid = project_points_to_image(
                sampled_target_means, frame_data['extrinsics'], frame_data['intrinsics'], (orig_H, orig_W)
            )

            vis_conf = (sampled_vis[0, 0, 0, :] * sampled_conf[0, 0, 0, :]).detach().cpu().numpy()
            valid = proj_valid.detach().cpu().numpy() & (vis_conf > 0.1)

            if valid.sum() > 0:
                errors = torch.sqrt(((projected_2d - tracked_pts) ** 2).sum(dim=-1)).detach().cpu().numpy()
                all_errors.extend(errors[valid].tolist())
                all_vis_conf.extend(vis_conf[valid].tolist())

        if all_errors:
            # Error histogram
            axes2[0].hist(all_errors, bins=30, edgecolor='black', alpha=0.7)
            axes2[0].axvline(np.mean(all_errors), color='red', linestyle='--', label=f'Mean: {np.mean(all_errors):.2f}px')
            axes2[0].axvline(np.median(all_errors), color='green', linestyle='--', label=f'Median: {np.median(all_errors):.2f}px')
            axes2[0].set_xlabel('Error (pixels)')
            axes2[0].set_ylabel('Count')
            axes2[0].set_title('Tracking vs Projection Error Distribution')
            axes2[0].legend()

            # Error vs confidence scatter
            axes2[1].scatter(all_vis_conf, all_errors, alpha=0.5, s=20)
            axes2[1].set_xlabel('Visibility × Confidence')
            axes2[1].set_ylabel('Error (pixels)')
            axes2[1].set_title('Error vs Tracking Confidence')

        plt.tight_layout()
        save_path2 = os.path.join(output_dir, f"tracking_stats_cam{int(cam_id)}_batch{b}.png")
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved tracking statistics to {save_path2}")
