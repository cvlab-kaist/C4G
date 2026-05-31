# WAN VACE Training & DAVIS Evaluation

This document describes the training and evaluation entry points used in this workspace.

## Paths

### Training
- Training code:
  - `/music-3d-shared-disk/user/KAIST/MK/foundation_models/DiffSynth-Studio-ref_keyframes_fixed/examples/wanvideo/model_training/train_vace.py`
- Training script:
  - `/music-3d-shared-disk/user/KAIST/MK/foundation_models/DiffSynth-Studio-ref_keyframes_fixed/examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh`

### DAVIS Evaluation
- Evaluation code:
  - `/music-3d-shared-disk/user/KAIST/MK/foundation_models/DiffSynth-Studio-ref_keyframes_fixed/examples/wanvideo/evaluation/evaluation_davis.py`
- Evaluation script:
  - `/music-3d-shared-disk/user/KAIST/MK/foundation_models/DiffSynth-Studio-ref_keyframes_fixed/examples/wanvideo/evaluation/evaulation_davis.sh`

## Checkpoint

- LoRA checkpoint used for evaluation:
  - `/music-3d-shared-disk/user/KAIST/MK/preprocessing/c4g/checkpoints/resume6-val-step-51960.safetensors`

## Training

Run training with:

```bash
bash /music-3d-shared-disk/user/KAIST/MK/foundation_models/DiffSynth-Studio-ref_keyframes_fixed/examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh
```

## Evaluation (DAVIS)

Run DAVIS evaluation with:

```bash
bash /music-3d-shared-disk/user/KAIST/MK/foundation_models/DiffSynth-Studio-ref_keyframes_fixed/examples/wanvideo/evaluation/evaulation_davis.sh
```

## Acknowledgements

We sincerely thank the great work Wan-Video, and DiffSynth-Studio for their inspiring work and contributions to the 3D and video generation community.
