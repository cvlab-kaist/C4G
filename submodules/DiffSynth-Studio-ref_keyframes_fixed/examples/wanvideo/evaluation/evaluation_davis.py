import argparse
import datetime as _dt
import json
import os
import re
import ssl
import sys
from dataclasses import dataclass

import cv2
import imageio
import lpips
import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb

DEFAULT_PRECOMPUTED_ROOT = (
    "/path/to/c4g_precomputed_davis"
)

DAVIS_ROOTS = [
    "/path/to/DAVIS_2017/JPEGImages/480p",
    "/path/to/DAVIS_2019/JPEGImages/480p",
]

WAN_H, WAN_W = 480, 832

DEFAULT_WAN_VACE_DIR = "/path/to/Wan2.1-VACE-1.3B"
DEFAULT_WAN_TOKENIZER = "/path/to/Wan2.1-VACE-1.3B/google/umt5-xxl"
DEFAULT_QWEN_VL = "/path/to/Qwen3-VL-8B-Instruct"



# Scene info parsing

@dataclass
class SceneInfo:
    start: int
    end: int
    gap: int
    target_indices: list  # all t_idx values present in rendered_by_view

    @property
    def context_indices(self):
        return list(range(self.start, self.end + 1, self.gap))

    @property
    def target_indices_between(self):
        ctx = self.context_indices
        return [t for c0, c1 in zip(ctx[:-1], ctx[1:]) for t in range(c0 + 1, c1)]


def parse_scene_info(scene_dir: str) -> SceneInfo:
    """Parse start/end/gap and available t_idx list from rendered_by_view/view_0000/."""
    view_dir = os.path.join(scene_dir, "rendered_by_view", "view_0000")
    if not os.path.isdir(view_dir):
        raise FileNotFoundError(f"No rendered_by_view/view_0000 found in {scene_dir}")

    pat = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)\.png$")
    starts, ends, gaps, t_idxs = set(), set(), set(), []
    for fname in os.listdir(view_dir):
        m = pat.match(fname)
        if m:
            starts.add(int(m.group(1)))
            ends.add(int(m.group(2)))
            gaps.add(int(m.group(3)))
            t_idxs.append(int(m.group(4)))

    if not t_idxs:
        raise FileNotFoundError(f"No {'{start}_{end}_{gap}_{t_idx}.png'} files in {view_dir}")

    if len(starts) != 1 or len(ends) != 1:
        # 여러 (start, end) 세트가 섞인 경우 — 파일 수가 가장 많은 세트 선택
        from collections import defaultdict, Counter
        group_count = Counter()
        group_tidxs = defaultdict(list)
        pat2 = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)\.png$")
        for fname in os.listdir(view_dir):
            m2 = pat2.match(fname)
            if m2:
                key = (int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
                group_count[key] += 1
                group_tidxs[key].append(int(m2.group(4)))
        best = max(group_count, key=group_count.get)
        print(f"  [warn] Multiple start/end sets in {view_dir}: {set(group_count.keys())} → using {best}")
        start, end, gap = best
        t_idxs = group_tidxs[best]
    else:
        start, end, gap = starts.pop(), ends.pop(), gaps.pop()

    return SceneInfo(
        start=start, end=end, gap=gap,
        target_indices=sorted(t_idxs),
    )


def find_davis_root(scene_name: str) -> str:
    for root in DAVIS_ROOTS:
        if os.path.isdir(os.path.join(root, scene_name)):
            return root
    raise FileNotFoundError(
        f"Scene '{scene_name}' not found in any DAVIS root: {DAVIS_ROOTS}"
    )


def load_davis_frame(davis_root: str, scene_name: str, frame_idx: int,
                     out_h: int = WAN_H, out_w: int = WAN_W) -> np.ndarray:
    """Load a DAVIS frame (BGR) and resize to (out_h, out_w)."""
    path = os.path.join(davis_root, scene_name, f"{frame_idx:05d}.jpg")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"DAVIS frame not found: {path}")
    if img.shape[:2] != (out_h, out_w):
        img = cv2.resize(img, (out_w, out_h))
    return img


def load_c4g_frame(scene_dir: str, info: SceneInfo, t_idx: int) -> np.ndarray:
    """Load a C4G rendered frame (BGR) from rendered_by_view/view_0000/."""
    fname = f"{info.start}_{info.end}_{info.gap}_{t_idx}.png"
    path = os.path.join(scene_dir, "rendered_by_view", "view_0000", fname)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"C4G frame not found: {path}")
    return img



# WAN helpers (copied verbatim from evaluation_davis_wan_sequence.py)
def pad_to_4k1(video: torch.Tensor, idx: torch.Tensor):
    t = video.shape[0]
    if (t - 1) % 4 == 0:
        return video, idx
    padded_len = ((t - 1) // 4 + 1) * 4 + 1
    pad_n = padded_len - t
    video = torch.cat([video, video[-1:].repeat(pad_n, *([1] * (video.dim() - 1)))], dim=0)
    if idx.shape[0] >= 2:
        step = int((idx[-1] - idx[-2]).item())
        if step <= 0:
            step = 1
    else:
        step = 1
    last_v = int(idx[-1].item())
    pad_idx = torch.tensor(
        [last_v + step * (i + 1) for i in range(pad_n)],
        dtype=idx.dtype, device=idx.device,
    )
    idx = torch.cat([idx, pad_idx], dim=0)
    return video, idx


def load_wan_pipeline(
    diffsynth_path: str,
    lora_path: str,
    *,
    vace_dir: str = DEFAULT_WAN_VACE_DIR,
    tokenizer_path: str = DEFAULT_WAN_TOKENIZER,
    qwen_path: str = DEFAULT_QWEN_VL,
    lora_rank: int = 128,
    device: str = "cuda",
):
    import json as _json
    import sys as _sys

    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")
    if not os.path.isdir(vace_dir):
        raise FileNotFoundError(f"VACE model dir not found: {vace_dir}")
    print(f"[wan] tokenizer_path={tokenizer_path}", flush=True)
    print(f"[wan] vace_dir={vace_dir}", flush=True)

    for _p in list(_sys.path):
        if _p != diffsynth_path and _p.endswith("DiffSynth-Studio-ref_keyframes_fixed"):
            _sys.path.remove(_p)
    if diffsynth_path not in _sys.path:
        _sys.path.insert(0, diffsynth_path)
    for _mod_name in list(_sys.modules):
        if _mod_name.startswith("examples.wanvideo") or _mod_name == "diffsynth" or _mod_name.startswith("diffsynth."):
            _sys.modules.pop(_mod_name, None)

    import examples.wanvideo.model_training.train_vace as _tv

    print(f"[wan] diffsynth_path={diffsynth_path}", flush=True)
    print(f"[wan] train_vace from: {_tv.__file__}", flush=True)

    wan_module = _tv.WanTrainingModule(
        model_paths=_json.dumps([
            f"{vace_dir}/diffusion_pytorch_model.safetensors",
            f"{vace_dir}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{vace_dir}/Wan2.1_VAE.pth",
        ]),
        tokenizer_path=tokenizer_path,
        lora_base_model="vace",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=lora_rank,
        lora_checkpoint=lora_path,
        device=device,
        task="sft",
    )
    wan_module.eval()
    captioner = _tv.QwenVLCaptioner(qwen_path, device=device)
    return wan_module, captioner


def build_wan_input(
    ctx_imgs_01, ctx_idx, target_rgb_01, target_idx, target_gt_01, *, ref_pos_offset=500
):
    gt_video_view1 = ctx_imgs_01
    gt_video_idx1 = ctx_idx
    context_video_keyframes = ctx_imgs_01
    context_idx_orig = ctx_idx.clone()

    rendered_video, rendered_video_idx = pad_to_4k1(target_rgb_01, target_idx)
    target_indices_orig = target_idx.clone()
    gt_video_view2, _ = pad_to_4k1(target_gt_01.clone(), target_idx.clone())

    gt_video = torch.cat([gt_video_view1, gt_video_view2], dim=0)
    vace_video = torch.cat([gt_video_view1, rendered_video], dim=0)
    vace_video_idx = torch.cat([gt_video_idx1 + ref_pos_offset, rendered_video_idx], dim=0)
    vace_mask = torch.cat([
        torch.zeros(len(gt_video_view1), dtype=torch.float32),
        torch.ones(len(rendered_video), dtype=torch.float32),
    ], dim=0)

    return {
        "gt_video": gt_video,
        "vace_video": vace_video,
        "vace_video_idx": vace_video_idx,
        "vace_video_mask": vace_mask,
        "context_video_keyframes": context_video_keyframes,
        "context_idx_orig": context_idx_orig,
        "target_indices_orig": target_indices_orig,
    }


@torch.no_grad()
def wan_refine(
    wan_module, captioner,
    ctx_imgs_01, ctx_idx, target_rgb_01, target_idx, target_gt_01,
    *, ref_pos_offset=500, num_steps=50, cfg_scale=5.0,
):
    pipe = wan_module.pipe
    data_wan = build_wan_input(
        ctx_imgs_01, ctx_idx, target_rgb_01, target_idx, target_gt_01,
        ref_pos_offset=ref_pos_offset,
    )
    pipe.scheduler.set_timesteps(1000, training=True)
    if "caption" not in data_wan:
        data_wan["caption"] = captioner.caption(data_wan["context_video_keyframes"])
    print(f"[wan caption] {data_wan['caption']}", flush=True)

    inputs = wan_module.get_pipeline_inputs(data_wan)
    inputs = wan_module.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
    for unit in pipe.units:
        inputs = pipe.unit_runner(unit, pipe, *inputs)
    inputs_shared, inputs_posi, inputs_nega = inputs

    pipe.scheduler.set_timesteps(num_steps, training=False)
    input_latents = inputs_shared["input_latents"]
    latents = torch.randn_like(input_latents)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for t in pipe.scheduler.timesteps:
        ts = t.to(dtype=pipe.torch_dtype, device=pipe.device).unsqueeze(0)
        inputs_shared["latents"] = latents
        np_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=ts)
        np_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=ts)
        latents = pipe.scheduler.step(np_nega + cfg_scale * (np_posi - np_nega), t, latents)

    t_lat_v1 = int(data_wan["context_idx_orig"].shape[0])
    d1 = torch.cat([
        pipe.vae.decode(latents[:, :, i:i + 1], device=pipe.device)[0]
        for i in range(t_lat_v1)
    ], dim=1)
    d2 = pipe.vae.decode(latents[:, :, t_lat_v1:], device=pipe.device)[0]
    n_target_frames_orig = int(data_wan["target_indices_orig"].shape[0])

    def _to_uint8(t):
        return ((t.float() + 1) * 127.5).clamp(0, 255).permute(1, 2, 3, 0).cpu().numpy().astype("uint8")

    return _to_uint8(d1), _to_uint8(d2[:, :n_target_frames_orig]), data_wan["caption"]



# Metrics
def compute_metrics_batch(gt_list, pred_list, lpips_fn, device, label):
    if len(gt_list) != len(pred_list):
        raise ValueError(f"compute_metrics_batch[{label}]: len(gt)={len(gt_list)} != len(pred)={len(pred_list)}")
    p_vals, s_vals = [], []
    gt_tensors, pred_tensors = [], []
    for gt_bgr, pred_bgr in zip(gt_list, pred_list):
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
        if gt_rgb.shape[:2] != pred_rgb.shape[:2]:
            pred_rgb = cv2.resize(pred_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]))
        p_vals.append(compute_psnr(gt_rgb, pred_rgb))
        s_vals.append(compute_ssim(gt_rgb, pred_rgb, channel_axis=2))
        gt_tensors.append(torch.from_numpy(gt_rgb).permute(2, 0, 1).float() / 255.0)
        pred_tensors.append(torch.from_numpy(pred_rgb).permute(2, 0, 1).float() / 255.0)
    gt_batch = torch.stack(gt_tensors).to(device) * 2 - 1
    pred_batch = torch.stack(pred_tensors).to(device) * 2 - 1
    l_vals = lpips_fn(gt_batch, pred_batch).squeeze().detach().cpu().numpy()
    if np.ndim(l_vals) == 0:
        l_vals = np.array([float(l_vals)])
    metrics = {
        "psnr": float(np.mean(p_vals)),
        "ssim": float(np.mean(s_vals)),
        "lpips": float(np.mean(l_vals)),
    }
    print(f"  {label}: PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}")
    return metrics



# Video save helpers


def save_video(frames_bgr, path, fps=10):
    if not frames_bgr:
        print(f"  Warning: skip empty video {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    imageio.mimwrite(path, frames_rgb, fps=fps, quality=8)
    h, w = frames_bgr[0].shape[:2]
    print(f"  Saved: {path} ({len(frames_bgr)} frames, {w}x{h}, {fps}fps)")


# Main processing
@torch.no_grad()
def process_scene(
    scene_dir: str,
    scene_name: str,
    out_dir: str,
    wan_module,
    captioner,
    lpips_fn,
    device,
    fps: int,
    ref_pos_offset: int,
    wan_num_steps: int,
    wan_cfg_scale: float,
    save_local_outputs: bool,
):
    info = parse_scene_info(scene_dir)
    ctx_indices = info.context_indices
    tgt_indices = info.target_indices
    between_indices = info.target_indices_between

    # v2_fixed training schedule: reference/view1 keyframes are sampled at stride 4
    # on the early part of the window (up to start+28).
    context_end = min(info.end, info.start + 28)
    ctx_indices_wan = [t for t in tgt_indices if info.start <= t <= context_end and (t - info.start) % 4 == 0]
    if len(ctx_indices_wan) == 0:
        raise ValueError(
            f"{scene_name}: no WAN context indices at stride-4 in [{info.start}, {context_end}] from targets."
        )

    print(
        f"  [indices] start={info.start} end={info.end} gap={info.gap}  "
        f"ctx_src({len(ctx_indices)})={ctx_indices[:3]}..{ctx_indices[-1]}  "
        f"ctx_wan({len(ctx_indices_wan)})={ctx_indices_wan[:3]}..{ctx_indices_wan[-1]}  "
        f"target({len(tgt_indices)})  between({len(between_indices)})"
    )

    # Verify all C4G target frames exist
    for t in tgt_indices:
        fname = f"{info.start}_{info.end}_{info.gap}_{t}.png"
        p = os.path.join(scene_dir, "rendered_by_view", "view_0000", fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing C4G frame: {p}")

    davis_root = find_davis_root(scene_name)
    print(f"  [davis] {davis_root}/{scene_name}")

    # Load WAN reference context GT at stride-4 schedule.
    ctx_gt_bgr = [load_davis_frame(davis_root, scene_name, i) for i in ctx_indices_wan]

    # Load full-target GT from DAVIS (for metrics and WAN conditioning)
    tgt_gt_bgr = [load_davis_frame(davis_root, scene_name, i) for i in tgt_indices]

    # Load C4G rendered frames
    c4g_bgr = [load_c4g_frame(scene_dir, info, i) for i in tgt_indices]

    # Verify resolution consistency
    for name, frames in [("ctx_gt", ctx_gt_bgr), ("tgt_gt", tgt_gt_bgr), ("c4g", c4g_bgr)]:
        for i, f in enumerate(frames):
            assert f.shape[:2] == (WAN_H, WAN_W), \
                f"{name}[{i}] shape mismatch: {f.shape[:2]} != ({WAN_H},{WAN_W})"

    # between-frame positions within tgt_indices list
    between_pos_in_tgt = [tgt_indices.index(t) for t in between_indices]

    base_metrics_between = compute_metrics_batch(
        [tgt_gt_bgr[i] for i in between_pos_in_tgt],
        [c4g_bgr[i] for i in between_pos_in_tgt],
        lpips_fn, device, "C4G (target_between)",
    )
    base_metrics_all = compute_metrics_batch(
        tgt_gt_bgr,
        c4g_bgr,
        lpips_fn, device, "C4G (target_all)",
    )

    ctx_imgs_01 = torch.from_numpy(
        np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in ctx_gt_bgr], axis=0)
    ).float().div(255).permute(0, 3, 1, 2).contiguous().to(device)
    target_rgb_01 = torch.from_numpy(
        np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in c4g_bgr], axis=0)
    ).float().div(255).permute(0, 3, 1, 2).contiguous().to(device)
    target_gt_01 = torch.from_numpy(
        np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in tgt_gt_bgr], axis=0)
    ).float().div(255).permute(0, 3, 1, 2).contiguous().to(device)
    # 0-normalize: WAN은 0-based index로 학습됨 (MK 원본 스크립트와 동일하게)
    offset = info.start
    ctx_idx_t = torch.tensor([i - offset for i in ctx_indices_wan], dtype=torch.long, device=device)
    target_idx_t = torch.tensor([i - offset for i in tgt_indices], dtype=torch.long, device=device)

    pred_v1_uint8, pred_v2_uint8, caption = wan_refine(
        wan_module, captioner,
        ctx_imgs_01, ctx_idx_t,
        target_rgb_01, target_idx_t, target_gt_01,
        ref_pos_offset=ref_pos_offset,
        num_steps=wan_num_steps,
        cfg_scale=wan_cfg_scale,
    )

    wan_tgt_bgr = [cv2.cvtColor(pred_v2_uint8[i], cv2.COLOR_RGB2BGR) for i in range(pred_v2_uint8.shape[0])]
    wan_tgt_bgr = [
        cv2.resize(f, (WAN_W, WAN_H)) if f.shape[:2] != (WAN_H, WAN_W) else f
        for f in wan_tgt_bgr
    ]
    if len(wan_tgt_bgr) != len(tgt_indices):
        raise ValueError(
            f"{scene_name}: WAN view2 output length {len(wan_tgt_bgr)} != target {len(tgt_indices)}"
        )

    wan_view1_bgr = [cv2.cvtColor(pred_v1_uint8[i], cv2.COLOR_RGB2BGR) for i in range(pred_v1_uint8.shape[0])]
    wan_view1_bgr = [
        cv2.resize(f, (WAN_W, WAN_H)) if f.shape[:2] != (WAN_H, WAN_W) else f
        for f in wan_view1_bgr
    ]

    wan_between_bgr = [wan_tgt_bgr[i] for i in between_pos_in_tgt]
    wan_metrics_between = compute_metrics_batch(
        [tgt_gt_bgr[i] for i in between_pos_in_tgt],
        wan_between_bgr,
        lpips_fn, device, "WAN (target_between)",
    )
    wan_metrics_all = compute_metrics_batch(
        tgt_gt_bgr,
        wan_tgt_bgr,
        lpips_fn, device, "WAN (target_all)",
    )

    # Combined side-by-side: c4g | gt | wan
    combined_between = [
        np.concatenate([c4g_bgr[i], tgt_gt_bgr[i], wan_between_bgr[idx]], axis=1)
        for idx, i in enumerate(between_pos_in_tgt)
    ]
    combined_full = [
        np.concatenate([c4g_bgr[i], tgt_gt_bgr[i], wan_tgt_bgr[i]], axis=1)
        for i in range(len(tgt_indices))
    ]
    combined_view1 = [
        np.concatenate([ctx_gt_bgr[i], wan_view1_bgr[i]], axis=1)
        for i in range(min(len(ctx_gt_bgr), len(wan_view1_bgr)))
    ]

    if save_local_outputs:
        os.makedirs(out_dir, exist_ok=True)
        save_video(wan_tgt_bgr, os.path.join(out_dir, "target_wan_full.mp4"), fps=fps)
        save_video(wan_between_bgr, os.path.join(out_dir, "target_wan.mp4"), fps=fps)
        save_video(wan_view1_bgr, os.path.join(out_dir, "context_wan_view1.mp4"), fps=fps)
        save_video(combined_view1, os.path.join(out_dir, "view1_ctx_wan.mp4"), fps=fps)
        save_video(combined_between, os.path.join(out_dir, "view2_between_c4g_gt_wan.mp4"), fps=fps)
        save_video(combined_full, os.path.join(out_dir, "view2_full_c4g_gt_wan.mp4"), fps=fps)
        with open(os.path.join(out_dir, f"{scene_name}_metrics.txt"), "w") as f:
            metrics_to_save = {
                **{f"c4g_between_{k}": v for k, v in base_metrics_between.items()},
                **{f"wan_between_{k}": v for k, v in wan_metrics_between.items()},
                **{f"c4g_all_{k}": v for k, v in base_metrics_all.items()},
                **{f"wan_all_{k}": v for k, v in wan_metrics_all.items()},
            }
            for k, v in metrics_to_save.items():
                f.write(f"{k}={v:.6f}\n")

    if wandb.run is not None:
        wandb.log({
            "test/wan_between_psnr": wan_metrics_between["psnr"],
            "test/wan_between_ssim": wan_metrics_between["ssim"],
            "test/wan_between_lpips": wan_metrics_between["lpips"],
            "test/c4g_between_psnr": base_metrics_between["psnr"],
            "test/c4g_between_ssim": base_metrics_between["ssim"],
            "test/c4g_between_lpips": base_metrics_between["lpips"],
            "test/wan_all_psnr": wan_metrics_all["psnr"],
            "test/wan_all_ssim": wan_metrics_all["ssim"],
            "test/wan_all_lpips": wan_metrics_all["lpips"],
            "test/c4g_all_psnr": base_metrics_all["psnr"],
            "test/c4g_all_ssim": base_metrics_all["ssim"],
            "test/c4g_all_lpips": base_metrics_all["lpips"],
            f"test/view2_between/{scene_name}": wandb.Video(
                np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in combined_between], axis=0).transpose(0, 3, 1, 2),
                fps=fps, format="mp4",
                caption=f"c4g | gt | wan | {caption}",
            ),
        })

    return {
        "c4g_between_psnr": base_metrics_between["psnr"], "c4g_between_ssim": base_metrics_between["ssim"], "c4g_between_lpips": base_metrics_between["lpips"],
        "wan_between_psnr": wan_metrics_between["psnr"], "wan_between_ssim": wan_metrics_between["ssim"], "wan_between_lpips": wan_metrics_between["lpips"],
        "c4g_all_psnr": base_metrics_all["psnr"], "c4g_all_ssim": base_metrics_all["ssim"], "c4g_all_lpips": base_metrics_all["lpips"],
        "wan_all_psnr": wan_metrics_all["psnr"], "wan_all_ssim": wan_metrics_all["ssim"], "wan_all_lpips": wan_metrics_all["lpips"],
        "caption": caption,
    }


def main():
    parser = argparse.ArgumentParser(description="WAN DAVIS eval for 480x832 C4G outputs")
    parser.add_argument("--precomputed_root", default=DEFAULT_PRECOMPUTED_ROOT)
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Scene names (without davis_ prefix). Default: auto-discover.")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--save_local_outputs", action="store_true")

    parser.add_argument("--diffsynth_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--ref_pos_offset", type=int, default=500)
    parser.add_argument("--wan_num_steps", type=int, default=50)
    parser.add_argument("--wan_cfg_scale", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--output_root", type=str,
                        default="./outputs/wan_eval")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="직접 출력 폴더 지정 (재시작용). 지정하면 output_root/lora/timestamp 무시.")

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    device = torch.device(args.device)

    print("Loading LPIPS...")
    _orig = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    ssl._create_default_https_context = _orig

    print("Loading WAN pipeline...")
    wan_module, captioner = load_wan_pipeline(
        diffsynth_path=args.diffsynth_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        device=str(device),
    )

    if not os.path.isdir(args.precomputed_root):
        raise FileNotFoundError(f"Precomputed root not found: {args.precomputed_root}")

    if args.scenes is None:
        scenes = sorted(
            d[len("davis_"):]
            for d in os.listdir(args.precomputed_root)
            if d.startswith("davis_") and os.path.isdir(os.path.join(args.precomputed_root, d))
        )
        print(f"[scenes] auto-discovered {len(scenes)}: {scenes[:5]}...")
    else:
        scenes = args.scenes

    if args.output_dir:
        eval_dir = args.output_dir
    else:
        lora_suffix = os.path.basename(args.lora_path)
        if lora_suffix.endswith(".safetensors"):
            lora_suffix = lora_suffix[:-len(".safetensors")]
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_dir = os.path.join(args.output_root, lora_suffix, ts)
    print(f"[output] {eval_dir}", flush=True)

    all_metrics = {}
    all_captions = {}
    for scene_name in tqdm(scenes, desc="Scenes", total=len(scenes)):
        scene_dir = os.path.join(args.precomputed_root, f"davis_{scene_name}")
        if not os.path.isdir(scene_dir):
            print(f"Skipping {scene_name}: not found ({scene_dir})")
            continue
        print(f"\n=== {scene_name} ===")
        try:
            scene_out_dir = os.path.join(eval_dir, f"davis_{scene_name}")
            m = process_scene(
                scene_dir=scene_dir,
                scene_name=scene_name,
                out_dir=scene_out_dir,
                wan_module=wan_module,
                captioner=captioner,
                lpips_fn=lpips_fn,
                device=device,
                fps=args.fps,
                ref_pos_offset=args.ref_pos_offset,
                wan_num_steps=args.wan_num_steps,
                wan_cfg_scale=args.wan_cfg_scale,
                save_local_outputs=args.save_local_outputs,
            )
            all_captions[f"davis_{scene_name}"] = m.pop("caption", "")
            all_metrics[scene_name] = m
        except Exception as e:
            import traceback
            print(f"  FAILED on {scene_name}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 72)
    print(f"[summary] scenes={len(all_metrics)}")
    print(f"{'Scene':<22} {'C4G_B_PSNR':>10} {'WAN_B_PSNR':>10} {'C4G_A_PSNR':>10} {'WAN_A_PSNR':>10}")
    print("-" * 72)
    c4g_pb, wan_pb, c4g_pa, wan_pa = [], [], [], []
    c4g_sb, wan_sb, c4g_sa, wan_sa = [], [], [], []
    c4g_lb, wan_lb, c4g_la, wan_la = [], [], [], []
    for sname, m in all_metrics.items():
        print(f"{sname:<22} {m['c4g_between_psnr']:>10.2f} {m['wan_between_psnr']:>10.2f} {m['c4g_all_psnr']:>10.2f} {m['wan_all_psnr']:>10.2f}")
        c4g_pb.append(m["c4g_between_psnr"]); wan_pb.append(m["wan_between_psnr"])
        c4g_pa.append(m["c4g_all_psnr"]); wan_pa.append(m["wan_all_psnr"])
        c4g_sb.append(m["c4g_between_ssim"]); wan_sb.append(m["wan_between_ssim"])
        c4g_sa.append(m["c4g_all_ssim"]); wan_sa.append(m["wan_all_ssim"])
        c4g_lb.append(m["c4g_between_lpips"]); wan_lb.append(m["wan_between_lpips"])
        c4g_la.append(m["c4g_all_lpips"]); wan_la.append(m["wan_all_lpips"])
    if c4g_pb:
        print("-" * 72)
        print(f"{'Mean':<22} {np.mean(c4g_pb):>10.2f} {np.mean(wan_pb):>10.2f} {np.mean(c4g_pa):>10.2f} {np.mean(wan_pa):>10.2f}")
        print(
            f"[mean-ssim] between: c4g={np.mean(c4g_sb):.4f} wan={np.mean(wan_sb):.4f} | "
            f"all: c4g={np.mean(c4g_sa):.4f} wan={np.mean(wan_sa):.4f}"
        )
        print(
            f"[mean-lpips] between: c4g={np.mean(c4g_lb):.4f} wan={np.mean(wan_lb):.4f} | "
            f"all: c4g={np.mean(c4g_la):.4f} wan={np.mean(wan_la):.4f}"
        )
    print("=" * 72)

    if args.save_local_outputs and all_metrics:
        import csv
        csv_path = os.path.join(eval_dir, "metrics_summary.csv")
        captions_path = os.path.join(eval_dir, "captions.json")
        os.makedirs(eval_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "scene",
                "c4g_between_psnr", "wan_between_psnr", "c4g_between_ssim", "wan_between_ssim", "c4g_between_lpips", "wan_between_lpips",
                "c4g_all_psnr", "wan_all_psnr", "c4g_all_ssim", "wan_all_ssim", "c4g_all_lpips", "wan_all_lpips",
            ])
            w.writeheader()
            for sname, m in all_metrics.items():
                w.writerow({"scene": sname, **m})
        with open(captions_path, "w") as f:
            json.dump(all_captions, f, indent=2, ensure_ascii=False)
        print(f"[csv] {csv_path}")
        print(f"[json] {captions_path}")


if __name__ == "__main__":
    main()
