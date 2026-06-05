"""Render qualitative results on DAVIS with pose alignment.

Uses DA3 to estimate pseudo camera poses, then applies Umeyama Sim(3) pose
alignment between the first-pass and second-pass DA3 predictions to improve
rendering quality. Outputs GT | Ours side-by-side videos per scene.

Usage (from C4G/ directory):
    python scripts/render_qualitative_davis.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --output_dir ./outputs/qualitative_davis

    # Specify scenes:
    python scripts/render_qualitative_davis.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --scenes blackswan dog camel
"""

import argparse
import os
import sys

import cv2
import imageio
import lpips
import numpy as np
import ssl
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loss.loss_tracking import project_points_to_image
from src.misc.cam_utils import camera_normalization


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DAVIS_ROOT = "path/to/davis"

# RE10K mean intrinsics (normalized, for square images)
# fx=fy~0.8767, cx=cy=0.5
RE10K_INTRINSIC = np.array([
    [0.8767, 0.0,    0.5],
    [0.0,    0.8767, 0.5],
    [0.0,    0.0,    1.0],
], dtype=np.float32)


def put_label(img, text, font_scale=0.8, thickness=2, margin=10):
    """Overlay a white text label with dark background on the top-left of a BGR image (in-place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Draw background rectangle
    cv2.rectangle(img, (0, 0), (tw + 2 * margin, th + 2 * margin + baseline), (0, 0, 0), -1)
    # Draw text
    cv2.putText(img, text, (margin, th + margin), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# DA3 pose estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_poses_da3(
    image_paths: list[str],
    num_frames: int,
    device: torch.device,
    da3_model_name: str = "depth-anything/DA3-LARGE-1.1",
    process_res: int = 504,
):
    """Estimate camera extrinsics for DAVIS frames using DA3.

    Returns:
        extrinsics_c2w: (N, 4, 4) float32 numpy array of c2w poses
    """
    from depth_anything_3.api import DepthAnything3

    print(f"  Loading DA3 model ({da3_model_name})...")
    da3 = DepthAnything3.from_pretrained(da3_model_name)
    da3 = da3.to(device).eval()

    paths = image_paths[:num_frames]
    print(f"  Running DA3 pose estimation on {len(paths)} frames (process_res={process_res})...")
    prediction = da3.inference(paths, process_res=process_res)

    # Extrinsics: w2c
    w2c = prediction.extrinsics  # numpy
    if w2c.shape[-2] == 3:
        pad = np.zeros((len(w2c), 4, 4), dtype=np.float32)
        pad[:, :3, :] = w2c
        pad[:, 3, 3] = 1.0
        w2c = pad

    c2w = np.linalg.inv(w2c).astype(np.float32)

    translations = c2w[:, :3, 3]
    dists = np.linalg.norm(translations - translations[0:1], axis=1)
    print(f"  DA3 translation range: {dists.min():.4f} - {dists.max():.4f}")

    del da3
    torch.cuda.empty_cache()

    return c2w


# ---------------------------------------------------------------------------
# DAVIS scene loading
# ---------------------------------------------------------------------------

def load_davis_scene(davis_root: str, scene_name: str, resolution: str = "480p"):
    frame_dir = os.path.join(davis_root, "JPEGImages", resolution, scene_name)
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    image_paths = [os.path.join(frame_dir, f) for f in frames]
    return image_paths


def load_and_crop_image(path: str, render_size: int):
    """Load image, center-crop to square, resize to render_size. Returns RGB uint8."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    crop_sz = min(w, h)
    left = (w - crop_sz) // 2
    top = (h - crop_sz) // 2
    img = img.crop((left, top, left + crop_sz, top + crop_sz))
    img = img.resize((render_size, render_size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Batch preparation with pose alignment
# ---------------------------------------------------------------------------

def prepare_batch_with_pose_align(
    image_paths: list[str],
    context_indices: list[int],
    target_indices: list[int],
    extrinsics_c2w: np.ndarray,
    target_c2w: np.ndarray | None = None,
    device: torch.device = torch.device("cuda"),
):
    """Build batch for DAVIS with DA3 poses and RE10K intrinsics.

    Uses baseline normalization and relative pose normalization.

    Args:
        image_paths: all frame paths
        context_indices: indices of context frames (original frame indices)
        target_indices: indices of target frames (original frame indices)
        extrinsics_c2w: (n_ctx, 4, 4) c2w poses from DA3 (one per context frame)
        target_c2w: (n_tgt, 4, 4) c2w poses from DA3 for target frames (optional)
        device: torch device

    Returns:
        batch dict, scale factor
    """
    n_ctx = len(context_indices)
    n_tgt = len(target_indices)

    # RE10K intrinsics (same for all frames, already for square images)
    ctx_intr = torch.tensor(RE10K_INTRINSIC, dtype=torch.float32).unsqueeze(0).expand(n_ctx, -1, -1).clone()

    ext = torch.tensor(extrinsics_c2w, dtype=torch.float32)  # (n_ctx, 4, 4) c2w

    # Baseline normalization
    a = ext[0, :3, 3]
    diff = ext[1:, :3, 3] - a
    norms = diff.norm(dim=1)
    scale = norms.max().item()
    if scale > 1e-6:
        ext[:, :3, 3] /= scale
    else:
        print("  Warning: near-zero baseline, skipping normalization")
        scale = 1.0

    # Relative pose (first context = identity)
    ref = ext[0:1]
    ext = camera_normalization(ref, ext)

    # Normalize target poses with the same scale and reference
    if target_c2w is not None and n_tgt > 0:
        tgt_ext = torch.tensor(target_c2w, dtype=torch.float32)
        if scale > 1e-6:
            tgt_ext[:, :3, 3] /= scale
        tgt_ext = camera_normalization(ref, tgt_ext)
        tgt_intr = torch.tensor(RE10K_INTRINSIC, dtype=torch.float32).unsqueeze(0).expand(n_tgt, -1, -1).clone()
    else:
        tgt_ext = ext[:n_tgt]
        tgt_intr = ctx_intr[:n_tgt]

    # Load context images -> normalize -> resize to 224
    ctx_imgs = []
    for idx in context_indices:
        img_pil = Image.fromarray(
            load_and_crop_image(image_paths[idx], 224)
        )
        arr = np.asarray(img_pil, dtype=np.float32) / 255.0
        ctx_imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    ctx_images = torch.stack(ctx_imgs).unsqueeze(0)  # (1, Vc, 3, 224, 224)
    ctx_images = (ctx_images - 0.5) / 0.5

    near_val = 0.01
    far_val = 100.0

    batch = {
        "context": {
            "extrinsics": ext.unsqueeze(0).to(device),
            "intrinsics": ctx_intr.unsqueeze(0).to(device),
            "image": ctx_images.to(device),
            "near": torch.full((1, n_ctx), near_val, device=device),
            "far": torch.full((1, n_ctx), far_val, device=device),
            "index": torch.tensor(context_indices, device=device).unsqueeze(0),
            "camera": torch.zeros(1, n_ctx, dtype=torch.long, device=device),
        },
        "target": {
            "extrinsics": tgt_ext.unsqueeze(0).to(device),
            "intrinsics": tgt_intr.unsqueeze(0).to(device),
            "near": torch.full((1, n_tgt), near_val, device=device),
            "far": torch.full((1, n_tgt), far_val, device=device),
            "index": torch.tensor(target_indices, device=device).unsqueeze(0),
            "camera": torch.zeros(1, n_tgt, dtype=torch.long, device=device),
        },
        "scene": ["davis"],
    }

    return batch, scale


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    """Load encoder + decoder from a Lightning checkpoint."""
    from src.model.encoder import get_encoder
    from src.model.encoder.encoder_vggt import EncoderVGGTCfg, OpacityMappingCfg
    from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
    from src.model.encoder.backbone.backbone_croco import BackboneCrocoCfg
    from src.model.decoder import get_decoder
    from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    encoder_state = {}
    decoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state[k[8:]] = v
        elif k.startswith("decoder."):
            decoder_state[k[8:]] = v

    num_gaussians = encoder_state.get("gaussian_tokens", torch.zeros(2048)).shape[0]

    backbone_cfg = BackboneCrocoCfg(name="vggt_multi", model="ViTLarge_BaseDecoder")
    gaussian_adapter_cfg = GaussianAdapterCfg(
        gaussian_scale_min=0.5, gaussian_scale_max=15.0, sh_degree=0, clamping=-1,
    )
    opacity_cfg = OpacityMappingCfg(initial=0.0, final=0.0, warm_up=1)
    encoder_cfg = EncoderVGGTCfg(
        name="vggt", d_feature=64, num_monocular_samples=32,
        backbone=backbone_cfg, gaussian_adapter=gaussian_adapter_cfg,
        opacity_mapping=opacity_cfg, num_gaussians=num_gaussians,
        timestamp_embedding=True, timestamp_embedding_type="sinusoidal",
        sinusoidal_embedding_dim=256,
    )

    encoder, _ = get_encoder(encoder_cfg)
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"Encoder missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    encoder = encoder.to(device).eval()

    decoder_cfg = DecoderSplattingCUDACfg(
        name="splatting_cuda", background_color=[0.0, 0.0, 0.0],
        make_scale_invariant=True, low_pass_filter=0.3,
    )
    decoder = get_decoder(decoder_cfg)
    if decoder_state:
        decoder.load_state_dict(decoder_state, strict=False)
    decoder = decoder.to(device).eval()

    return encoder, decoder


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def save_video(frames_bgr, path, fps=10):
    h, w = frames_bgr[0].shape[:2]
    writer = imageio.get_writer(path, fps=fps, codec="libx264")
    for f in frames_bgr:
        writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"  Saved video: {path} ({len(frames_bgr)} frames, {w}x{h}, {fps}fps)")


def save_image_grid(images_bgr, path, nrow=None):
    """Save a grid of images. images_bgr: list of same-sized BGR arrays."""
    n = len(images_bgr)
    if nrow is None:
        nrow = min(n, 8)
    ncol = (n + nrow - 1) // nrow
    h, w = images_bgr[0].shape[:2]
    grid = np.zeros((ncol * h, nrow * w, 3), dtype=np.uint8)
    for i, img in enumerate(images_bgr):
        r, c = divmod(i, nrow)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    cv2.imwrite(path, grid)
    print(f"  Saved grid: {path} ({nrow}x{ncol})")



# ---------------------------------------------------------------------------
# Main rendering
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_scene(
    encoder,
    decoder,
    lpips_fn,
    davis_root: str,
    output_dir: str,
    scene_name: str,
    resolution: str = "480p",
    num_context_frames: int = 16,
    gap: int = 2,
    render_size: int = 512,
    fps: int = 5,
    da3_process_res: int = 504,
    device: torch.device = torch.device("cuda"),
):
    """Render qualitative results for a single DAVIS scene.

    Pipeline:
    1. Load DAVIS frames
    2. Estimate DA3 poses for context frames
    3. Build batch with DA3 poses (baseline + relative normalization)
       -> run encoder -> get Gaussians
    4. Render from the same normalized poses, compare with GT
    5. Save videos (context, target, combined)
    """
    image_paths = load_davis_scene(davis_root, scene_name, resolution)
    T = len(image_paths)

    # Build context / target indices
    if gap > 0:
        context_indices = list(range(0, T, gap))[:num_context_frames]
    else:
        context_indices = list(range(min(num_context_frames, T)))

    # Target indices: frames between context frames
    target_indices_between = []
    for ci in range(len(context_indices) - 1):
        start, end = context_indices[ci], context_indices[ci + 1]
        for idx in range(start + 1, end):
            target_indices_between.append(idx)

    max_frame = max(context_indices) + 1
    n_target = len(target_indices_between)
    print(f"  T={T} frames, Context: {len(context_indices)}, Target: {n_target}, gap={gap}")
    print(f"  Context indices: {context_indices[:5]}...{context_indices[-3:]}")
    if n_target > 0:
        print(f"  Target indices: {target_indices_between[:5]}...{target_indices_between[-3:]}")

    # Step 1: Estimate DA3 poses for all frames up to max_frame
    ext_c2w = estimate_poses_da3(
        image_paths, max_frame, device, process_res=da3_process_res,
    )

    # Select poses for context and target frames
    ctx_c2w = ext_c2w[context_indices]  # (n_ctx, 4, 4)
    tgt_c2w = ext_c2w[target_indices_between] if n_target > 0 else None

    # Step 2: Build batch and run encoder (always use RE10K intrinsics)
    batch, scale = prepare_batch_with_pose_align(
        image_paths, context_indices, target_indices_between if n_target > 0 else context_indices,
        extrinsics_c2w=ctx_c2w,
        target_c2w=tgt_c2w,
        device=device,
    )

    # Get gaussians for context timestamps
    ctx_timestamps = torch.unique(batch["context"]["index"])
    # Also request gaussians for target timestamps
    if n_target > 0:
        tgt_timestamps = torch.unique(batch["target"]["index"])
        all_timestamps = torch.unique(torch.cat([ctx_timestamps, tgt_timestamps]))
    else:
        all_timestamps = ctx_timestamps
    gaussian_per_timestamp = encoder(
        batch["context"], global_step=0, target_timestamps=all_timestamps,
    )

    # Step 3: Render from the same normalized poses the encoder saw
    # (DA3 poses with baseline normalization + relative pose normalization)
    render_ext = batch["context"]["extrinsics"]   # (1, Vc, 4, 4)
    render_intr = batch["context"]["intrinsics"]   # (1, Vc, 3, 3)
    render_near = batch["context"]["near"]
    render_far = batch["context"]["far"]

    # Step 4: Render from each frame's camera pose
    os.makedirs(output_dir, exist_ok=True)
    scene_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    sorted_ts = sorted(ctx_timestamps.tolist())
    render_frames_bgr = []
    gt_frames_bgr = []

    psnr_vals, ssim_vals = [], []
    lpips_gt_tensors, lpips_pred_tensors = [], []

    for i, t_idx in enumerate(sorted_ts):
        gaussians = gaussian_per_timestamp[t_idx]

        # Render
        output = decoder.forward(
            gaussians,
            render_ext[:, i:i+1],
            render_intr[:, i:i+1],
            render_near[:, i:i+1],
            render_far[:, i:i+1],
            (render_size, render_size),
        )
        render_img = output.color[0, 0].clamp(0, 1)
        render_rgb = (render_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
        render_frames_bgr.append(render_bgr)

        # GT frame (center-cropped, resized)
        frame_idx = context_indices[i]
        gt_rgb = load_and_crop_image(image_paths[frame_idx], render_size)
        gt_bgr = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
        gt_frames_bgr.append(gt_bgr)

        # Metrics
        psnr_vals.append(compute_psnr(gt_rgb, render_rgb))
        ssim_vals.append(compute_ssim(gt_rgb, render_rgb, channel_axis=2))
        lpips_gt_tensors.append(torch.from_numpy(gt_rgb).permute(2, 0, 1).float() / 255.0)
        lpips_pred_tensors.append(torch.from_numpy(render_rgb).permute(2, 0, 1).float() / 255.0)

    # --- Metrics helper ---
    def compute_metrics_batch(gt_list, pred_list, label):
        p_vals, s_vals = [], []
        lg_tensors, lp_tensors = [], []
        for i in range(len(gt_list)):
            gt_rgb = cv2.cvtColor(gt_list[i], cv2.COLOR_BGR2RGB)
            pred_rgb = cv2.cvtColor(pred_list[i], cv2.COLOR_BGR2RGB)
            p_vals.append(compute_psnr(gt_rgb, pred_rgb))
            s_vals.append(compute_ssim(gt_rgb, pred_rgb, channel_axis=2))
            lg_tensors.append(torch.from_numpy(gt_rgb).permute(2, 0, 1).float() / 255.0)
            lp_tensors.append(torch.from_numpy(pred_rgb).permute(2, 0, 1).float() / 255.0)
        lg_batch = torch.stack(lg_tensors).to(device) * 2 - 1
        lp_batch = torch.stack(lp_tensors).to(device) * 2 - 1
        l_vals = lpips_fn(lg_batch, lp_batch).squeeze().cpu().numpy()
        if l_vals.ndim == 0:
            l_vals = np.array([l_vals.item()])
        del lg_batch, lp_batch
        torch.cuda.empty_cache()
        m_p = float(np.mean(p_vals))
        m_s = float(np.mean(s_vals))
        m_l = float(np.mean(l_vals))
        print(f"  {label} Metrics: PSNR={m_p:.2f}, SSIM={m_s:.4f}, LPIPS={m_l:.4f}")
        return {"psnr": m_p, "ssim": m_s, "lpips": m_l}

    ctx_metrics = compute_metrics_batch(gt_frames_bgr, render_frames_bgr, "Context")

    # Step 5: Save outputs (moving cam versions kept for reference)

    # Moving-cam video (no labels, for reference)
    ours_video_path = os.path.join(scene_dir, f"{scene_name}_ours_moving.mp4")
    save_video(render_frames_bgr, ours_video_path, fps=fps)

    # Static camera render (identity pose, all timestamps)
    print("  Rendering static camera video...")
    static_ext = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
    static_intr = render_intr[:, 0:1]
    static_near = render_near[:, 0:1]
    static_far = render_far[:, 0:1]
    static_frames_bgr = []
    for t_idx in sorted_ts:
        gaussians = gaussian_per_timestamp[t_idx]
        output = decoder.forward(
            gaussians, static_ext, static_intr,
            static_near, static_far, (render_size, render_size),
        )
        render_img = output.color[0, 0].clamp(0, 1)
        render_rgb = (render_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        static_frames_bgr.append(cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR))
    static_video_path = os.path.join(scene_dir, f"{scene_name}_static_cam.mp4")
    save_video(static_frames_bgr, static_video_path, fps=fps)

    # GT video (no label)
    save_video(gt_frames_bgr, os.path.join(scene_dir, f"{scene_name}_gt.mp4"), fps=fps)

    # Ours video (static cam, no label)
    save_video(static_frames_bgr, os.path.join(scene_dir, f"{scene_name}_ours.mp4"), fps=fps)

    # --- Target frames (between context frames) ---
    tgt_metrics = None
    if n_target > 0:
        tgt_render_ext = batch["target"]["extrinsics"]   # (1, n_tgt, 4, 4)
        tgt_render_intr = batch["target"]["intrinsics"]   # (1, n_tgt, 3, 3)
        tgt_render_near = batch["target"]["near"]
        tgt_render_far = batch["target"]["far"]
        sorted_tgt_ts = sorted(tgt_timestamps.tolist())

        # Render target moving camera frames
        print("  Rendering target moving camera frames...")
        target_render_frames_bgr = []
        for i, t_idx in enumerate(sorted_tgt_ts):
            gaussians = gaussian_per_timestamp[t_idx]
            output = decoder.forward(
                gaussians,
                tgt_render_ext[:, i:i+1],
                tgt_render_intr[:, i:i+1],
                tgt_render_near[:, i:i+1],
                tgt_render_far[:, i:i+1],
                (render_size, render_size),
            )
            render_img = output.color[0, 0].clamp(0, 1)
            render_rgb = (render_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            target_render_frames_bgr.append(cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR))

        # Render target static camera frames
        print("  Rendering target static camera frames...")
        target_static_frames_bgr = []
        for t_idx in sorted_tgt_ts:
            gaussians = gaussian_per_timestamp[t_idx]
            output = decoder.forward(
                gaussians, static_ext, static_intr,
                static_near, static_far, (render_size, render_size),
            )
            render_img = output.color[0, 0].clamp(0, 1)
            render_rgb = (render_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            target_static_frames_bgr.append(cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR))

        # Target GT frames
        target_gt_frames_bgr = []
        for idx in target_indices_between:
            gt_rgb = load_and_crop_image(image_paths[idx], render_size)
            target_gt_frames_bgr.append(cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))

        # Target metrics
        tgt_metrics = compute_metrics_batch(target_gt_frames_bgr, target_render_frames_bgr, "Target")

        # Save target videos
        save_video(target_gt_frames_bgr, os.path.join(scene_dir, f"{scene_name}_target_gt.mp4"), fps=fps)
        save_video(target_render_frames_bgr, os.path.join(scene_dir, f"{scene_name}_target_ours_moving.mp4"), fps=fps)
        save_video(target_static_frames_bgr, os.path.join(scene_dir, f"{scene_name}_target_ours.mp4"), fps=fps)

    # --- Combined (context + target) videos in temporal order ---
    if n_target > 0:
        # Build index-to-frame maps
        ctx_gt_map = {context_indices[i]: gt_frames_bgr[i] for i in range(len(context_indices))}
        ctx_moving_map = {context_indices[i]: render_frames_bgr[i] for i in range(len(context_indices))}
        ctx_static_map = {context_indices[i]: static_frames_bgr[i] for i in range(len(context_indices))}
        tgt_gt_map = {target_indices_between[i]: target_gt_frames_bgr[i] for i in range(n_target)}
        tgt_moving_map = {target_indices_between[i]: target_render_frames_bgr[i] for i in range(n_target)}
        tgt_static_map = {target_indices_between[i]: target_static_frames_bgr[i] for i in range(n_target)}

        all_indices = sorted(set(context_indices) | set(target_indices_between))
        all_gt = [ctx_gt_map.get(i, tgt_gt_map.get(i)) for i in all_indices]
        all_moving = [ctx_moving_map.get(i, tgt_moving_map.get(i)) for i in all_indices]
        all_static = [ctx_static_map.get(i, tgt_static_map.get(i)) for i in all_indices]

        save_video(all_gt, os.path.join(scene_dir, f"{scene_name}_all_gt.mp4"), fps=fps)
        save_video(all_moving, os.path.join(scene_dir, f"{scene_name}_all_ours_moving.mp4"), fps=fps)
        save_video(all_static, os.path.join(scene_dir, f"{scene_name}_all_ours.mp4"), fps=fps)

    result = {"psnr": ctx_metrics["psnr"], "ssim": ctx_metrics["ssim"], "lpips": ctx_metrics["lpips"]}
    if tgt_metrics is not None:
        result["target_psnr"] = tgt_metrics["psnr"]
        result["target_ssim"] = tgt_metrics["ssim"]
        result["target_lpips"] = tgt_metrics["lpips"]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render qualitative DAVIS results with pose alignment"
    )
    parser.add_argument("--checkpoint", default="path/to/c4g_reconstructor.ckpt")
    parser.add_argument("--davis_root", default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="DAVIS scene names (default: all scenes)")
    parser.add_argument("--resolution", default="480p", choices=["480p", "1080p"])
    parser.add_argument("--output_dir", default="./outputs/video_davis")
    parser.add_argument("--num_context_frames", type=int, default=16)
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--render_size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--da3_process_res", type=int, default=504)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading model...")
    encoder, decoder = load_model(args.checkpoint, device)

    print("Loading LPIPS model...")
    _orig_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    ssl._create_default_https_context = _orig_ctx

    if args.scenes is None:
        jpeg_dir = os.path.join(args.davis_root, "JPEGImages", args.resolution)
        args.scenes = sorted(os.listdir(jpeg_dir))

    all_metrics = {}
    for scene_name in args.scenes:
        scene_dir = os.path.join(args.davis_root, "JPEGImages", args.resolution, scene_name)
        if not os.path.isdir(scene_dir):
            print(f"Skipping {scene_name}: not found at {scene_dir}")
            continue

        print(f"\nProcessing {scene_name}...")
        metrics = render_scene(
            encoder, decoder, lpips_fn,
            args.davis_root, args.output_dir, scene_name,
            resolution=args.resolution,
            num_context_frames=args.num_context_frames,
            gap=args.gap,
            render_size=args.render_size,
            fps=args.fps,
            da3_process_res=args.da3_process_res,
            device=device,
        )
        all_metrics[scene_name] = metrics

    # Summary table — Context
    print("\n" + "=" * 60)
    print("Context frames:")
    print(f"{'Scene':<20} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("-" * 60)
    psnr_all, ssim_all, lpips_all = [], [], []
    for scene_name, m in all_metrics.items():
        print(f"{scene_name:<20} {m['psnr']:>8.2f} {m['ssim']:>8.4f} {m['lpips']:>8.4f}")
        psnr_all.append(m["psnr"])
        ssim_all.append(m["ssim"])
        lpips_all.append(m["lpips"])
    if psnr_all:
        print("-" * 60)
        print(f"{'Mean':<20} {np.mean(psnr_all):>8.2f} {np.mean(ssim_all):>8.4f} {np.mean(lpips_all):>8.4f}")
    print("=" * 60)

    # Summary table — Target
    has_target = any("target_psnr" in m for m in all_metrics.values())
    if has_target:
        print("\nTarget frames (between context):")
        print(f"{'Scene':<20} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
        print("-" * 60)
        tgt_p, tgt_s, tgt_l = [], [], []
        for scene_name, m in all_metrics.items():
            if "target_psnr" in m:
                print(f"{scene_name:<20} {m['target_psnr']:>8.2f} {m['target_ssim']:>8.4f} {m['target_lpips']:>8.4f}")
                tgt_p.append(m["target_psnr"])
                tgt_s.append(m["target_ssim"])
                tgt_l.append(m["target_lpips"])
        if tgt_p:
            print("-" * 60)
            print(f"{'Mean':<20} {np.mean(tgt_p):>8.2f} {np.mean(tgt_s):>8.4f} {np.mean(tgt_l):>8.4f}")
        print("=" * 60)

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\nMax GPU memory allocated: {max_mem:.2f} GB")

    print("\nDone.")


if __name__ == "__main__":
    main()
