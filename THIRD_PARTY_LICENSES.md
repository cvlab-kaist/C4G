# Third-Party Licenses

This repository includes or builds on the following third-party code. Each component remains under its own
license, which takes precedence over the MIT license in [LICENSE](LICENSE) for that component. Several of
them permit **non-commercial / research use only**; using the full training pipeline or the released weights
is therefore subject to those terms.

| Component | Location in this repository | License |
|---|---|---|
| VGGT (Meta) | `src/model/encoder/backbone/vggt/` | VGGT License (see [facebookresearch/vggt](https://github.com/facebookresearch/vggt)) |
| MoGe (Microsoft) | `src/model/moge/` | MIT (see [microsoft/MoGe](https://github.com/microsoft/MoGe)) |
| DINOv2 (Meta) | `src/model/moge/dinov2/` | Apache-2.0 (see [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)) |
| CroCo / DUSt3R utilities (Naver) | `src/model/encoder/backbone/croco/misc.py` | CC BY-NC-SA 4.0, non-commercial |
| CoWTracker (Meta) | `submodules/cowtracker/` | FAIR Noncommercial Research License ([LICENSE](submodules/cowtracker/LICENSE)) |
| Differential Gaussian Rasterization (Inria / MPII) | `submodules/diff_gaussian_rasterization_w_pose/` | Gaussian-Splatting License, non-commercial ([LICENSE.md](submodules/diff_gaussian_rasterization_w_pose/LICENSE.md)) |
| GLM | `submodules/diff_gaussian_rasterization_w_pose/third_party/glm` | MIT / Happy Bunny License |
| DiffSynth-Studio | `submodules/DiffSynth-Studio/` | Apache-2.0 ([LICENSE](submodules/DiffSynth-Studio/LICENSE)) |

Models downloaded at run time are governed by their own terms: MoGe-2 (`Ruicheng/moge-2-vitl-normal`),
CoWTracker weights, torchvision RAFT, and Wan2.1 for the video generation module. The released T4GS weights are
initialized from C3G and VGGT and are provided for research purposes.
