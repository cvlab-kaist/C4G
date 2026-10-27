"""Lossless export of renderer Gaussians to standard INRIA 3DGS PLY files.

The model stores renderer-ready attributes (linear scale, sigmoid opacity and
an xyzw quaternion).  The original 3D Gaussian Splatting PLY convention stores
the inverse-activation parameters (log scale, logit opacity) and a wxyz
quaternion.  This module performs exactly those representation changes.  It
does not prune, reorder, downsample or transform Gaussian positions.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


_PLY_SCALAR_DTYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "ushort": "<u2",
    "uint16": "<u2",
    "short": "<i2",
    "int16": "<i2",
    "uint": "<u4",
    "uint32": "<u4",
    "int": "<i4",
    "int32": "<i4",
}


def _single_batch(tensor: Tensor, batch_index: int, name: str) -> Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"{name} must include batch and Gaussian dimensions")
    if not 0 <= batch_index < tensor.shape[0]:
        raise IndexError(
            f"batch_index={batch_index} is invalid for {name} with shape {tuple(tensor.shape)}"
        )
    return tensor[batch_index].detach().to(dtype=torch.float32, device="cpu").contiguous()


def _finite(name: str, tensor: Tensor) -> None:
    if not torch.isfinite(tensor).all():
        count = int((~torch.isfinite(tensor)).sum())
        raise ValueError(f"{name} contains {count} NaN/Inf values")


def _extract_renderer_attributes(
    gaussians: Any,
    batch_index: int,
) -> dict[str, Tensor]:
    if gaussians.scales is None or gaussians.rotations is None:
        raise ValueError(
            "Standard 3DGS PLY export requires explicit scales and rotations; "
            "refusing to infer them ambiguously from covariance matrices."
        )

    attrs = {
        "means": _single_batch(gaussians.means, batch_index, "means"),
        "covariances": _single_batch(
            gaussians.covariances, batch_index, "covariances"
        ),
        "harmonics": _single_batch(gaussians.harmonics, batch_index, "harmonics"),
        "opacities": _single_batch(gaussians.opacities, batch_index, "opacities"),
        "scales": _single_batch(gaussians.scales, batch_index, "scales"),
        "rotations_xyzw": _single_batch(
            gaussians.rotations, batch_index, "rotations"
        ),
    }
    num_gaussians = attrs["means"].shape[0]
    expected_shapes = {
        "means": (num_gaussians, 3),
        "covariances": (num_gaussians, 3, 3),
        "opacities": (num_gaussians,),
        "scales": (num_gaussians, 3),
        "rotations_xyzw": (num_gaussians, 4),
    }
    for name, expected in expected_shapes.items():
        if tuple(attrs[name].shape) != expected:
            raise ValueError(
                f"Unexpected {name} shape {tuple(attrs[name].shape)}; expected {expected}"
            )
    if attrs["harmonics"].ndim != 3 or tuple(attrs["harmonics"].shape[:2]) != (
        num_gaussians,
        3,
    ):
        raise ValueError(
            "harmonics must have shape [gaussian, 3, d_sh], got "
            f"{tuple(attrs['harmonics'].shape)}"
        )
    d_sh = attrs["harmonics"].shape[-1]
    degree = math.isqrt(d_sh) - 1
    if (degree + 1) ** 2 != d_sh:
        raise ValueError(f"SH coefficient count must be a square, got d_sh={d_sh}")
    for name, tensor in attrs.items():
        _finite(name, tensor)
    if not torch.all(attrs["scales"] > 0):
        raise ValueError("Renderer scales must be strictly positive for log-scale export")
    if not torch.all((attrs["opacities"] >= 0) & (attrs["opacities"] <= 1)):
        raise ValueError("Renderer opacities must be in the closed interval [0, 1]")
    if not torch.all(attrs["rotations_xyzw"].norm(dim=-1) > 0):
        raise ValueError("Renderer rotations contain a zero quaternion")
    return attrs


def _standard_3dgs_columns(attrs: Mapping[str, Tensor]) -> tuple[list[str], np.ndarray]:
    means = attrs["means"]
    harmonics = attrs["harmonics"]
    opacities = attrs["opacities"]
    scales = attrs["scales"]
    rotations_xyzw = attrs["rotations_xyzw"]
    num_gaussians = means.shape[0]

    names = ["x", "y", "z", "nx", "ny", "nz"]
    parts = [means, torch.zeros_like(means)]

    names.extend(f"f_dc_{index}" for index in range(3))
    parts.append(harmonics[:, :, 0])

    num_rest = 3 * (harmonics.shape[-1] - 1)
    if num_rest:
        # INRIA stores the non-DC coefficients channel-major: all R, then G, then B.
        names.extend(f"f_rest_{index}" for index in range(num_rest))
        parts.append(harmonics[:, :, 1:].reshape(num_gaussians, num_rest))

    names.append("opacity")
    # A finite PLY cannot represent logit(0/1). Clamp only saturated float32
    # endpoints by one epsilon; sigmoid round-trip error stays <= one epsilon.
    opacity_eps = torch.finfo(opacities.dtype).eps
    finite_opacities = opacities.clamp(opacity_eps, 1 - opacity_eps)
    parts.append(torch.logit(finite_opacities).unsqueeze(-1))

    names.extend(f"scale_{index}" for index in range(3))
    parts.append(scales.log())

    # Model/renderer: xyzw (SciPy). INRIA 3DGS PLY: wxyz.
    rotations_wxyz = rotations_xyzw[:, [3, 0, 1, 2]]
    names.extend(f"rot_{index}" for index in range(4))
    parts.append(rotations_wxyz)

    columns = torch.cat(parts, dim=-1)
    _finite("serialized PLY columns", columns)
    return names, columns.numpy().astype("<f4", copy=False)


def write_3dgs_ply(
    path: Path | str,
    gaussians: Any,
    *,
    batch_index: int = 0,
) -> dict[str, Any]:
    """Write one renderer Gaussian scene as binary little-endian 3DGS PLY."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    attrs = _extract_renderer_attributes(gaussians, batch_index)
    names, columns = _standard_3dgs_columns(attrs)
    dtype = np.dtype([(name, "<f4") for name in names])
    vertices = np.empty(columns.shape[0], dtype=dtype)
    for index, name in enumerate(names):
        vertices[name] = columns[:, index]

    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment T4GS renderer Gaussians serialized in INRIA 3DGS convention",
        f"element vertex {len(vertices)}",
        *(f"property float {name}" for name in names),
        "end_header",
        "",
    ]
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(vertices.tobytes(order="C"))

    return {
        "path": str(path),
        "num_gaussians": len(vertices),
        "num_sh_coefficients_per_channel": attrs["harmonics"].shape[-1],
        "sh_degree": math.isqrt(attrs["harmonics"].shape[-1]) - 1,
        "properties": names,
        "format": "binary_little_endian 1.0",
        "opacity_endpoint_clamp_count": int(
            ((attrs["opacities"] == 0) | (attrs["opacities"] == 1)).sum()
        ),
        "opacity_endpoint_clamp_epsilon": torch.finfo(torch.float32).eps,
    }


def read_3dgs_ply(path: Path | str) -> tuple[dict[str, Any], np.ndarray]:
    """Read a scalar-property binary little-endian PLY for export verification."""
    path = Path(path)
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii").rstrip("\n")
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = None
        count = None
        properties: list[tuple[str, str]] = []
        in_vertex = False
        header_lines = [first]
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"PLY header has no end_header: {path}")
            line = raw.decode("ascii").rstrip("\n")
            header_lines.append(line)
            tokens = line.split()
            if tokens[:1] == ["format"]:
                fmt = tokens[1]
            elif tokens[:2] == ["element", "vertex"]:
                count = int(tokens[2])
                in_vertex = True
            elif tokens[:1] == ["element"]:
                in_vertex = False
            elif tokens[:1] == ["property"] and in_vertex:
                if len(tokens) != 3 or tokens[1] == "list":
                    raise ValueError("Only scalar vertex properties are supported")
                properties.append((tokens[2], tokens[1]))
            elif line == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"Expected binary_little_endian PLY, got {fmt}")
        if count is None:
            raise ValueError("PLY has no vertex element")
        try:
            dtype = np.dtype(
                [(name, _PLY_SCALAR_DTYPES[type_name]) for name, type_name in properties]
            )
        except KeyError as error:
            raise ValueError(f"Unsupported PLY scalar type: {error.args[0]}") from error
        vertices = np.fromfile(handle, dtype=dtype, count=count)
        if len(vertices) != count:
            raise ValueError(f"Expected {count} vertices, read {len(vertices)}")
    return {
        "format": fmt,
        "num_gaussians": count,
        "properties": [name for name, _ in properties],
        "header": header_lines,
    }, vertices


def _build_covariance_xyzw(scales: Tensor, rotations_xyzw: Tensor) -> Tensor:
    x, y, z, w = rotations_xyzw.unbind(dim=-1)
    two_s = 2 / rotations_xyzw.square().sum(dim=-1)
    matrix = torch.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    scaled = matrix @ torch.diag_embed(scales)
    return scaled @ scaled.transpose(-1, -2)


def validate_3dgs_ply(
    path: Path | str,
    gaussians: Any | None = None,
    *,
    batch_index: int = 0,
    atol: float = 2e-6,
) -> dict[str, Any]:
    """Validate structure, finiteness and optional round-trip model agreement."""
    header, vertices = read_3dgs_ply(path)
    property_names = header["properties"]
    required = {
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required.difference(property_names))
    if missing:
        raise ValueError(f"Missing standard 3DGS properties: {missing}")
    matrix = np.column_stack([vertices[name] for name in property_names])
    if not np.isfinite(matrix).all():
        raise ValueError(f"PLY contains {int((~np.isfinite(matrix)).sum())} NaN/Inf values")

    result: dict[str, Any] = {
        **header,
        "finite": True,
        "roundtrip_matches_model": None,
    }
    if gaussians is None:
        return result

    attrs = _extract_renderer_attributes(gaussians, batch_index)
    if header["num_gaussians"] != attrs["means"].shape[0]:
        raise ValueError(
            f"Gaussian count mismatch: PLY={header['num_gaussians']}, "
            f"model={attrs['means'].shape[0]}"
        )

    def stack(names: list[str]) -> Tensor:
        return torch.from_numpy(np.column_stack([vertices[name] for name in names]).copy())

    recovered_means = stack(["x", "y", "z"])
    recovered_opacities = stack(["opacity"]).sigmoid().squeeze(-1)
    recovered_scales = stack([f"scale_{index}" for index in range(3)]).exp()
    recovered_wxyz = stack([f"rot_{index}" for index in range(4)])
    recovered_xyzw = recovered_wxyz[:, [1, 2, 3, 0]]
    recovered_dc = stack([f"f_dc_{index}" for index in range(3)])

    rest_names = sorted(
        (name for name in property_names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if rest_names:
        recovered_rest = stack(rest_names).reshape(len(vertices), 3, -1)
        recovered_harmonics = torch.cat([recovered_dc.unsqueeze(-1), recovered_rest], -1)
    else:
        recovered_harmonics = recovered_dc.unsqueeze(-1)
    recovered_covariances = _build_covariance_xyzw(recovered_scales, recovered_xyzw)

    comparisons = {
        "position_max_abs_error": (recovered_means - attrs["means"]).abs().max().item(),
        "opacity_max_abs_error": (
            recovered_opacities - attrs["opacities"]
        ).abs().max().item(),
        "scale_max_abs_error": (recovered_scales - attrs["scales"]).abs().max().item(),
        "rotation_xyzw_max_abs_error": (
            recovered_xyzw - attrs["rotations_xyzw"]
        ).abs().max().item(),
        "sh_max_abs_error": (
            recovered_harmonics - attrs["harmonics"]
        ).abs().max().item(),
        "covariance_max_abs_error": (
            recovered_covariances - attrs["covariances"]
        ).abs().max().item(),
    }
    result.update(comparisons)
    result["quaternion_norm_min"] = float(recovered_xyzw.norm(dim=-1).min())
    result["quaternion_norm_max"] = float(recovered_xyzw.norm(dim=-1).max())
    result["roundtrip_matches_model"] = all(value <= atol for value in comparisons.values())
    if not result["roundtrip_matches_model"]:
        raise ValueError(f"PLY round-trip exceeds atol={atol}: {comparisons}")
    return result


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "unknown"


def export_gaussian_sequence(
    gaussians_per_timestamp: Mapping[int | float, Any],
    sequence_dir: Path | str,
    *,
    source_sample: str,
    checkpoint_step: int | None,
    batch_index: int = 0,
    extra_metadata: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Export sorted timestamps as frame_0000.ply, frame_0001.ply, ... ."""
    sequence_dir = Path(sequence_dir)
    timestamps = sorted(gaussians_per_timestamp)
    if not timestamps:
        raise ValueError("Cannot export an empty Gaussian sequence")
    expected_names = [f"frame_{index:04d}.ply" for index in range(len(timestamps))]
    if sequence_dir.exists():
        existing = sorted(path.name for path in sequence_dir.glob("frame_*.ply"))
        if existing:
            raise FileExistsError(
                f"Refusing to mix/overwrite an existing sequence at {sequence_dir}: "
                f"found {len(existing)} frame files"
            )
    sequence_dir.mkdir(parents=True, exist_ok=True)

    frame_records = []
    counts = []
    for frame_index, (timestamp, filename) in enumerate(zip(timestamps, expected_names)):
        gaussians = gaussians_per_timestamp[timestamp]
        path = sequence_dir / filename
        record = write_3dgs_ply(path, gaussians, batch_index=batch_index)
        if validate:
            checks = validate_3dgs_ply(path, gaussians, batch_index=batch_index)
            record["validation"] = {
                key: value
                for key, value in checks.items()
                if key not in {"header", "properties"}
            }
        record.update({"frame_index": frame_index, "timestamp": timestamp})
        frame_records.append(record)
        counts.append(record["num_gaussians"])

    metadata: dict[str, Any] = {
        "num_frames": len(timestamps),
        "timestamps": timestamps,
        "num_gaussians_per_frame": counts,
        "source_sample": source_sample,
        "checkpoint_step": checkpoint_step,
        "frame_filenames": expected_names,
        "coordinate_system": (
            "Model renderer/world coordinate system; identical for every frame; no export transform"
        ),
        "gaussian_row_order": (
            "Preserved exactly from model output; no sorting, filtering, pruning, or downsampling"
        ),
        "temporal_identity_by_row": len(set(counts)) == 1,
        "ply_convention": {
            "format": "INRIA/original 3DGS binary_little_endian PLY",
            "position": "renderer means, unchanged",
            "opacity": (
                "logit(renderer opacity); sigmoid on load; exact 0/1 endpoints are "
                "clamped inward by float32 epsilon to keep PLY values finite"
            ),
            "scale": "log(renderer linear anisotropic scale); exp on load",
            "rotation": "rot_0..3 are wxyz; renderer source is xyzw",
            "sh": (
                "f_dc_0..2 are RGB DC coefficients; f_rest_* are channel-major "
                "higher-order coefficients"
            ),
        },
        "validation_enabled": validate,
        "frames": frame_records,
    }
    if extra_metadata:
        metadata["source"] = dict(extra_metadata)
    metadata_path = sequence_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def validation_sequence_directory(
    root: Path | str,
    *,
    checkpoint_step: int,
    dataloader_index: int,
    batch_index: int,
    source_sample: str,
) -> Path:
    """Return a collision-resistant directory following the validation convention."""
    scene_name = (
        f"scene_{dataloader_index:03d}_{batch_index:04d}_{_safe_name(source_sample)}"
    )
    return (
        Path(root)
        / f"step_{checkpoint_step:06d}"
        / "val"
        / scene_name
        / "gaussian_sequence"
    )
