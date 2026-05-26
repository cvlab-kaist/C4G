import os
import torch
import numpy as np
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

num_patches = 256
patch_size_h = patch_size_w = int(np.sqrt(num_patches))

def depth_to_3d(depth_map, intrinsic_matrix):
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert pixel coordinates and depth values to 3D points
    x = (i - intrinsic_matrix[0, 2]) * depth_map / intrinsic_matrix[0, 0]
    y = (j - intrinsic_matrix[1, 2]) * depth_map / intrinsic_matrix[1, 1]
    z = depth_map
    
    points_3d = np.stack([x, y, z], axis=-1)
    return points_3d

def project_3d_to_2d(points_3d, intrinsic_matrix):
    # Convert 3D points to homogeneous coordinates
    projected_2d_hom = intrinsic_matrix @ points_3d.T
    # Convert from homogeneous coordinates to 2D image coordinates
    projected_2d = projected_2d_hom[:2, :] / projected_2d_hom[2, :]
    return projected_2d.T, projected_2d_hom[2, :]

def compute_optical_flow(depth1, depth2, pose1, pose2, intrinsic_matrix1, intrinsic_matrix2):
    # Input: All inputs as numpy arrays; convert torch tensors to numpy arrays if needed
    if isinstance(depth1, torch.Tensor):
        depth1 = depth1.cpu().numpy()
    if isinstance(depth2, torch.Tensor):
        depth2 = depth2.cpu().numpy()
    if isinstance(pose1, torch.Tensor):
        pose1 = pose1.cpu().numpy()
    if isinstance(pose2, torch.Tensor):
        pose2 = pose2.cpu().numpy()
    if isinstance(intrinsic_matrix1, torch.Tensor):
        intrinsic_matrix1 = intrinsic_matrix1.cpu().numpy()
    if isinstance(intrinsic_matrix2, torch.Tensor):
        intrinsic_matrix2 = intrinsic_matrix2.cpu().numpy()

    points_3d_frame1 = depth_to_3d(depth1, intrinsic_matrix1).reshape(-1, 3)
    points_3d_frame1_hom = np.concatenate([points_3d_frame1, np.ones((points_3d_frame1.shape[0], 1))], axis=1).T
    
    # Calculate the transformation matrix from frame 1 to frame 2
    transformation_matrix = (pose2) @ np.linalg.inv(pose1)
    points_3d_frame2_hom = transformation_matrix @ points_3d_frame1_hom
    points_3d_frame2 = (points_3d_frame2_hom[:3, :]).T

    points_2d_frame1, d1 = project_3d_to_2d(points_3d_frame1, intrinsic_matrix1)
    points_2d_frame2, d2 = project_3d_to_2d(points_3d_frame2, intrinsic_matrix2)

    # Compute optical flow vectors
    optical_flow = points_2d_frame2 - points_2d_frame1
    # Handle nan/inf values
    optical_flow = np.nan_to_num(optical_flow, nan=0.0, posinf=0.0, neginf=0.0)
    return optical_flow

def flow_to_dynamic_mask(
    flow: Float[Tensor, "TP H W 2"],
    depth: Float[Tensor, "T H W"],
    poses: Float[Tensor, "T 4 4"],
    intrinsics: Float[Tensor, "T 3 3"],
    threshold: float = 1.0,
) -> Float[Tensor, "T H W"]:
    """Compute dynamic mask from optical flow.

    Args:
        flow: Optical flow of shape (T-1, H, W, 2).
        depth: Depth maps of shape (T, H, W).
        poses: Camera poses of shape (T, 4, 4).
        intrinsics: Camera intrinsics of shape (T, 3, 3).
        threshold: Threshold for motion magnitude to consider as dynamic.
    Returns:
        Dynamic mask of shape (T, H, W), where 1 indicates dynamic regions.
        Last frame mask is set to -1 (invalid, no flow available).
    """
    assert flow.ndim == 4, "Flow must be a 4D tensor"
    t, h, w, _ = flow.shape
    intrinsics = intrinsics.clone()
    intrinsics[:, 0] *= w
    intrinsics[:, 1] *= h

    masks = []
    device = flow.device  # Get device from input tensor
    for i in range(t):
        # Load depth maps
        depth_map_frame1 = depth[i].cpu()  # Move to CPU for numpy conversion
        depth_map_frame2 = depth[i + 1].cpu()
        
        # Load camera intrinsics and poses
        intrinsic_matrix1, pose_frame1 = intrinsics[i].cpu(), poses[i].cpu()
        intrinsic_matrix2, pose_frame2 = intrinsics[i + 1].cpu(), poses[i + 1].cpu()

        # Compute optical flow
        optical_flow = compute_optical_flow(depth_map_frame1, depth_map_frame2, pose_frame1, pose_frame2, intrinsic_matrix1, intrinsic_matrix2)

        # Reshape the optical flow to the image dimensions and move to same device as ground truth flow
        height, width = depth_map_frame1.shape
        optical_flow_image = torch.from_numpy(optical_flow.reshape(height, width, 2)).float().to(device)

        # Load ground truth optical flow (already on device)
        gt_flow = flow[i]  # Assuming flow is pre-loaded and indexed by frame

        # Compute the error map (ensure same dtype)
        error_map = torch.linalg.norm(gt_flow.float() - optical_flow_image, dim=-1)
        # Handle nan/inf in error map
        error_map = torch.nan_to_num(error_map, nan=0.0, posinf=threshold*10, neginf=0.0)
        binary_error_map = error_map
        masks.append(binary_error_map)
        
    # Last frame has no flow, so fill with -1 (invalid marker)
    last_frame_mask = torch.full_like(masks[0], -1.0)
    masks.append(last_frame_mask)
    return torch.stack(masks, dim=0).float()

def precompute_spring_masks(data_root: str, output_dir: str, scene_list: list, original_image_shape: tuple = (1080, 1920), device: str = "cuda"):
    """Precompute and save masks for Spring scenes.
    
    Args:
        data_root: Path to Spring dataset root (e.g., /path/to/Spring/train)
        output_dir: Directory to save precomputed masks
        scene_list: List of scene names to process
        original_image_shape: Original image shape (H, W)
        device: Device to use ('cuda' or 'cpu')
    """
    import torchvision.transforms as tf
    from PIL import Image
    import h5py
    import cv2
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    to_tensor = tf.ToTensor()
    
    def readFlo5Flow(filename):
        with h5py.File(filename, "r") as f:
            return f["flow"][()]
    
    def readDsp5Disp(filename):
        with h5py.File(filename, "r") as f:
            return f["disparity"][()]
    
    def load_scene_data(scene_path):
        scene_id = os.path.basename(scene_path)
        
        extr_path = os.path.join(scene_path, "cam_data", "extrinsics.txt")
        intr_path = os.path.join(scene_path, "cam_data", "intrinsics.txt")
        extrinsics = np.loadtxt(extr_path).reshape(-1, 4, 4)
        intrinsics_original = np.loadtxt(intr_path).reshape(-1, 4)
        
        intrinsics = np.eye(3, dtype=np.float32)[None, :, :].repeat(extrinsics.shape[0], axis=0)
        intrinsics[:, 0, 0] = intrinsics_original[:, 0]
        intrinsics[:, 1, 1] = intrinsics_original[:, 1]
        intrinsics[:, 0, 2] = intrinsics_original[:, 2]
        intrinsics[:, 1, 2] = intrinsics_original[:, 3]
        intrinsics[:, 0, :] /= original_image_shape[1]
        intrinsics[:, 1, :] /= original_image_shape[0]
        
        # Load flows
        camera_name = "left"  # Extract camera name from flow directory
        
        def load_flow(i):
            flow_path = os.path.join(scene_path, "flow_FW_left", f"flow_FW_left_{i+1:04d}.flo5")
            flow = to_tensor(readFlo5Flow(flow_path)).to(torch.float32)
            return flow
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            flow_futures_with_idx = []
            for i in range(extrinsics.shape[0] - 1):
                flow_futures_with_idx.append(
                    (
                        i,
                        executor.submit(load_flow, i),
                    )
                )
            # Pre-allocate list with correct size to maintain order
            flows = [None] * (extrinsics.shape[0] - 1)
            for idx, future in flow_futures_with_idx:
                flows[idx] = future.result()
        
        flows = torch.stack(flows)
        flows = torch.einsum("tchw->thwc", flows)

        flows = flows[:, ::2, ::2] / 2
        flows = torch.nan_to_num(flows, nan=0.0, posinf=0.0, neginf=0.0)
        flows[..., 0] *= (patch_size_w * 14) / flows.shape[2]
        flows[..., 1] *= (patch_size_h * 14) / flows.shape[1]
        
        # Load depths
        def load_depth(i):
            disp_path = os.path.join(scene_path, "disp1_left", f"disp1_left_{i+1:04d}.dsp5")
            disp = to_tensor(readDsp5Disp(disp_path)).to(torch.float32)
            depth = torch.where(disp > 0, 0.065 * intrinsics[0, 0, 0] * disp.shape[1] / disp, torch.tensor(1e5, dtype=torch.float32))
            return depth
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            depth_futures_with_idx = []
            for i in range(extrinsics.shape[0]):
                depth_futures_with_idx.append(
                    (
                        i,
                        executor.submit(load_depth, i),
                    )
                )
            # Pre-allocate list with correct size to maintain order
            depths = [None] * extrinsics.shape[0]
            for idx, future in depth_futures_with_idx:
                depths[idx] = future.result()
        
        depths = torch.stack(depths).squeeze(1)
        depths = torch.nan_to_num(depths, nan=1e5, posinf=1e5, neginf=1e5)
        depths = torch.clamp(depths, min=0.01)  # Clamp to avoid division by very small values
        
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
        
        flows = F.interpolate(
            flows.permute(0, 3, 1, 2),
            size=(patch_size_h * 14, patch_size_w * 14),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        depths = F.interpolate(
            depths.unsqueeze(1),
            size=(patch_size_h * 14, patch_size_w * 14),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return flows, depths, extrinsics, intrinsics, scene_id, camera_name
    
    print(f"[{device}] Processing {len(scene_list)} scenes...")
    
    def process_scene(scene_path):
        scene_id = os.path.basename(scene_path)
        
        # Count cameras in source data
        camera_dirs = [d for d in os.listdir(scene_path) if d.startswith("flow_FW_left") or d.startswith("disp1_left")]
        num_cameras = len(set([d.split("_")[1] for d in camera_dirs]))  # Extract camera names (left, right, etc.)
        
        flows, depths, extrinsics, intrinsics, scene_id, camera_name = load_scene_data(scene_path)
        
        # Check if this camera already computed
        mask_file = output_dir / f"{scene_id}_masks.npy"
        if mask_file.exists():
            print(f"\n[{device}] Masks already exist for {scene_id}, skipping...")
            return scene_id
        
        # Move to device (GPU/CPU)
        flows = flows.to(device)
        depths = depths.to(device)
        extrinsics = extrinsics.to(device)
        intrinsics = intrinsics.to(device)
        
        # Compute masks on device
        masks = flow_to_dynamic_mask(flows, depths, extrinsics, intrinsics)
        # Move back to CPU for saving
        masks = masks.cpu().numpy()
        np.save(str(mask_file), masks)
        print(f"\n[{device}] Masks saved to {mask_file}")
        return scene_id
    
    for idx, scene_name in tqdm(enumerate(scene_list, 1), total=len(scene_list), desc=f"[{device}] Processing scenes"):
        print(f"[{device}] Processing {idx}/{len(scene_list)}: {scene_name}")
        process_scene(os.path.join(data_root, scene_name))


def load_masks_from_disk(mask_dir: str, scene_id: str, camera_name: str = "left") -> Float[Tensor, "T H W"]:
    """Load pre-computed dynamic masks from disk.

    Args:
        mask_dir: Directory containing saved masks.
        scene_id: Scene identifier for naming.
        camera_name: Camera name (default: "left").
    Returns:
        Dynamic mask of shape (T, H, W).
    """
    mask_dir = Path(mask_dir)
    scene_mask_dir = mask_dir / scene_id
    
    if not scene_mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found at {scene_mask_dir}")
    
    # Load NPY file
    mask_file = scene_mask_dir / "masks.npy"
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found at {mask_file}")
    
    masks = np.load(str(mask_file))
    masks = torch.from_numpy(masks).float()
    return masks


if __name__ == "__main__":
    num_gpus = 4  # Use single GPU for specific scenes
    data_root = "/path/to/Spring/train"
    output_dir = "/path/to/spring_dynamic_masks"
    os.makedirs(output_dir, exist_ok=True)

    # Process specific scenes only
    specific_scenes = None  # Change this to process different scenes
    all_scenes = sorted([s for s in os.listdir(data_root) if s in specific_scenes]) if specific_scenes is not None else sorted(os.listdir(data_root))
    print(all_scenes)
    
    # Split scenes across GPUs
    scenes_per_gpu = len(all_scenes) // num_gpus
    gpu_scenes = []
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * scenes_per_gpu
        if gpu_id == num_gpus - 1:
            # Last GPU gets remaining scenes
            end_idx = len(all_scenes)
        else:
            end_idx = start_idx + scenes_per_gpu
        gpu_scenes.append(all_scenes[start_idx:end_idx])
    
    # Process on multiple GPUs in parallel
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(
                precompute_spring_masks,
                data_root,
                output_dir,
                gpu_scenes[gpu_id],
                device=f"cuda:{gpu_id}"
            )
            for gpu_id in range(num_gpus)
        ]
        for future in futures:
            future.result()  # Wait for completion
