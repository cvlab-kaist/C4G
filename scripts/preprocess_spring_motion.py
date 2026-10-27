"""Precompute the per-frame motion supervision maps of the Spring dataset.

For every scene under ``<root>/train/<scene>`` (left camera) this writes, at
the pre-crop resolution the T4GS loader feeds to its crop shim
(``--height 224`` -> 224x398 for the 1920x1080 frames):

    <out>/<scene>/depth.npy       [T, h, w]    float16  metric depth from disp1, 0 = invalid
    <out>/<scene>/flow.npy        [T, h, w, 2] float16  forward flow (dx, dy) in output pixels
    <out>/<scene>/depth_next.npy  [T, h, w]    float16  depth of the correspondence at t+1 (disp2_FW)
    <out>/<scene>/mask.npy        [T, h, w]    uint8    1 = non-rigid (moving) pixel (rigidmap)
    <out>/<scene>/valid.npy       [T, h, w]    uint8    1 = flow / depth_next trustworthy

Conventions verified against the raw data (see memory note scene-flow-gt-sources):
the 4K arrays hold values in 1920x1080 pixel units, so they are subsampled by 2
before resizing; rigidmap True marks pixels whose motion is not explained by
the camera; matchmap non-black marks unmatched (occluded / out-of-frame) pixels.
The last frame has no forward data: its mask comes from the backward rigidmap
and its flow / depth_next / valid are zero.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

STEREO_BASELINE_M = 0.065


def read_h5(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f[key][()]


def resize_nearest(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x))
    if t.ndim == 2:
        t = t[None, None].float()
        return F.interpolate(t, size=shape, mode="nearest")[0, 0].numpy()
    t = t.permute(2, 0, 1)[None].float()
    return F.interpolate(t, size=shape, mode="nearest")[0].permute(1, 2, 0).numpy()


def process_scene(scene_dir: Path, out_dir: Path, height: int, overwrite: bool) -> str:
    scene = scene_dir.name
    scene_out = out_dir / scene
    if not overwrite and (scene_out / "valid.npy").exists():
        return f"{scene}: exists, skipped"
    scene_out.mkdir(parents=True, exist_ok=True)

    intr = np.loadtxt(scene_dir / "cam_data" / "intrinsics.txt").reshape(-1, 4)
    n = intr.shape[0]
    frame = np.asarray(Image.open(scene_dir / "frame_left" / "frame_left_0001.png"))
    src_h, src_w = frame.shape[:2]
    scale = height / src_h
    out_h, out_w = height, round(src_w * scale)
    sx, sy = out_w / src_w, out_h / src_h

    depth = np.zeros((n, out_h, out_w), np.float16)
    flow = np.zeros((n, out_h, out_w, 2), np.float16)
    depth_next = np.zeros((n, out_h, out_w), np.float16)
    mask = np.zeros((n, out_h, out_w), np.uint8)
    valid = np.zeros((n, out_h, out_w), np.uint8)

    for i in range(1, n + 1):
        fx = float(intr[i - 1, 0])
        disp1 = read_h5(scene_dir / "disp1_left" / f"disp1_left_{i:04d}.dsp5", "disparity")[::2, ::2].astype(np.float32)
        sky = np.asarray(Image.open(scene_dir / "maps" / "skymap_left" / f"skymap_left_{i:04d}.png"))[::2, ::2].astype(bool)
        d1_ok = np.isfinite(disp1) & (disp1 > 0) & ~sky
        d1 = np.where(d1_ok, STEREO_BASELINE_M * fx / np.maximum(disp1, 1e-6), 0.0)
        depth[i - 1] = resize_nearest(d1, (out_h, out_w)).astype(np.float16)

        if i < n:
            fl = read_h5(scene_dir / "flow_FW_left" / f"flow_FW_left_{i:04d}.flo5", "flow")[::2, ::2].astype(np.float32)
            disp2 = read_h5(scene_dir / "disp2_FW_left" / f"disp2_FW_left_{i:04d}.dsp5", "disparity")[::2, ::2].astype(np.float32)
            rigid = np.asarray(Image.open(scene_dir / "maps" / "rigidmap_FW_left" / f"rigidmap_FW_left_{i:04d}.png"))[::2, ::2].astype(bool)
            match = np.asarray(Image.open(scene_dir / "maps" / "matchmap_flow_FW_left" / f"matchmap_flow_FW_left_{i:04d}.png"))
            if match.ndim == 3:
                match = match.any(axis=-1)
            match = match.astype(bool)
            if match.shape != disp1.shape:
                match = resize_nearest(match.astype(np.float32), disp1.shape) > 0.5
            fl_ok = np.isfinite(fl).all(-1)
            d2_ok = np.isfinite(disp2) & (disp2 > 0)
            d2 = np.where(d2_ok, STEREO_BASELINE_M * fx / np.maximum(disp2, 1e-6), 0.0)
            fl = np.nan_to_num(fl, nan=0.0, posinf=0.0, neginf=0.0)
            v = d1_ok & fl_ok & d2_ok & ~match
            fl_out = resize_nearest(fl, (out_h, out_w))
            fl_out[..., 0] *= sx
            fl_out[..., 1] *= sy
            flow[i - 1] = fl_out.astype(np.float16)
            depth_next[i - 1] = resize_nearest(d2, (out_h, out_w)).astype(np.float16)
            valid[i - 1] = resize_nearest(v.astype(np.float32), (out_h, out_w)) > 0.5
            mask[i - 1] = resize_nearest(rigid.astype(np.float32), (out_h, out_w)) > 0.5
        else:
            rigid_bw = scene_dir / "maps" / "rigidmap_BW_left" / f"rigidmap_BW_left_{i:04d}.png"
            if rigid_bw.exists():
                rigid = np.asarray(Image.open(rigid_bw))[::2, ::2].astype(bool)
                mask[i - 1] = resize_nearest(rigid.astype(np.float32), (out_h, out_w)) > 0.5

    np.save(scene_out / "depth.npy", depth)
    np.save(scene_out / "flow.npy", flow)
    np.save(scene_out / "depth_next.npy", depth_next)
    np.save(scene_out / "mask.npy", mask)
    np.save(scene_out / "valid.npy", valid)
    return (
        f"{scene}: {n} frames -> {out_h}x{out_w}, dynamic {mask.mean():.3f}, "
        f"valid {valid[:-1].mean():.3f}, depth median {np.median(depth[depth > 0]):.2f} m"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Spring train split, e.g. /path/to/Spring/train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    scenes = sorted(args.scenes or [d.name for d in root.iterdir() if d.is_dir()])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_scene, root / s, out_dir, args.height, args.overwrite): s for s in scenes}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as e:  # keep going, report at the end
                print(f"{futures[fut]}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
