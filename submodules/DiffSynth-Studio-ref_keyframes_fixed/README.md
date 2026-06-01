# WAN VACE Training & DAVIS Evaluation

This document describes the training and evaluation entry points used in this workspace.

## Paths

### Training
- Training code:
  - `DiffSynth-Studio/examples/wanvideo/model_training/train_vace.py`
- Training script:
  - `DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh`

### DAVIS Evaluation
- Evaluation code:
  - `DiffSynth-Studio/examples/wanvideo/evaluation/evaluation_davis.py`
- Evaluation script:
  - `DiffSynth-Studio/examples/wanvideo/evaluation/evaulation_davis.sh`

## Checkpoint

- LoRA checkpoint used for evaluation (trained on 480x832):
  - [c4g_vdm_refinement.safetensors](https://huggingface.co/mungyeom011/C4G)

## Training

Run training with:

```bash
bash DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh
```

## Evaluation (DAVIS)

Run DAVIS evaluation with:

```bash
bash DiffSynth-Studio/examples/wanvideo/evaluation/evaulation_davis.sh
```

## Acknowledgements

We sincerely thank the great work Wan-Video, and DiffSynth-Studio for their inspiring work and contributions to the 3D and video generation community.
