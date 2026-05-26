"""Visualize camera centers from N×4×4 extrinsic matrices as a PLY file."""

import numpy as np


def save_camera_centers_ply(
    extrinsics: np.ndarray,
    path: str = "cameras.ply",
    w2c: bool = True,
):
    """Save camera centers as a colored point cloud PLY.

    Args:
        extrinsics: (N, 4, 4) array of extrinsic matrices.
        path: Output PLY file path.
        w2c: If True, extrinsics are world-to-camera (R|t where t = -R@C).
             If False, extrinsics are camera-to-world (the translation is the center).
    """
    assert extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4)
    N = extrinsics.shape[0]

    if w2c:
        # C = -R^T @ t
        R = extrinsics[:, :3, :3]  # (N, 3, 3)
        t = extrinsics[:, :3, 3]   # (N, 3)
        centers = -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)
    else:
        centers = extrinsics[:, :3, 3]

    # Color gradient: red (first) -> blue (last)
    colors = np.zeros((N, 3), dtype=np.uint8)
    if N > 1:
        alpha = np.linspace(0, 1, N)
        colors[:, 0] = (255 * (1 - alpha)).astype(np.uint8)  # R
        colors[:, 2] = (255 * alpha).astype(np.uint8)         # B
    else:
        colors[:, 0] = 255

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = centers[i]
            r, g, b = colors[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

    print(f"Saved {N} camera centers to {path}")