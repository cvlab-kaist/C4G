import torch, torch.nn as nn, os, argparse, accelerate, warnings, heapq
import numpy as np
import wandb
from datetime import datetime
import pytz
from PIL import Image
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
# Dataset class is selected at runtime via --dataset flag (see build_dataset()).
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# NOTE: 학습 시엔 데이터셋의 pre-computed caption (data["caption"]) 을 쓰지만,
#       eval (e.g., iphone) 에선 caption 이 없으므로 model_wrapper 가 이 클래스를
#       monkey-patch 해서 사용함. 따라서 클래스 정의는 유지.
class QwenVLCaptioner:
    """Qwen3-VL based video captioner. Generates text captions from video frames."""

    CAPTION_PREFIX = (
        "The following images are consecutive frames from a short video, shown in temporal order. "
        "Observe how objects change position between the frames to infer their motion. "
        "Describe the video in two or three natural sentences: include any objects present and, "
        "if they are moving across the frames, describe each object's motion direction "
        "(e.g., 'a red car drives to the left while a person walks forward'). "
        "Do NOT describe the camera (pan, zoom, static, etc.). "
        "Do NOT start with preambles like 'Based on the images' or 'Here is a description' — "
        "begin directly with the scene description."
    )

    def __init__(self, model_path="Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device="cuda", max_new_tokens=256):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=dtype,
        ).to(device).eval()
        self.model.requires_grad_(False)

        self.processor = AutoProcessor.from_pretrained(model_path)

    @torch.no_grad()
    def caption(self, video_frames: torch.Tensor, num_sample_frames=24) -> str:
        """
        video_frames: (N, C, H, W) float tensor in [0, 1].
        Returns a single caption string.
        """
        N = video_frames.shape[0]
        if N > num_sample_frames:
            indices = torch.linspace(0, N - 1, num_sample_frames).long()
        else:
            indices = torch.arange(N)
        sampled = video_frames[indices]  # (S, C, H, W)

        # Convert tensors to PIL images
        pil_images = []
        for frame in sampled:
            frame_np = (frame.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
            pil_images.append(Image.fromarray(frame_np))

        # Build message with interleaved images
        content = [{"type": "text", "text": self.CAPTION_PREFIX}]
        for img in pil_images:
            content.append({"type": "image", "image": img})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        return caption


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        # Caption 은 데이터셋에서 pre-computed 로 제공됨 (data["caption"]).
        # self.captioner = QwenVLCaptioner(
        #     "Qwen/Qwen3-VL-8B-Instruct", device=device,
        # )
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        # prompt = self.captioner.caption(data["gt_video"])
        # prompt = self.captioner.caption(data["context_video_keyframes"])
        prompt = data["caption"]
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": ""}
        vace_video = data["vace_video"]       # (T_view1 + T_view2, C, H, W) — unequal
        gt_video = data["gt_video"]            # (T_view1 + T_view2, C, H, W) — view1 keyframes + view2 padded
        # view1 length = len(context_idx_orig) (frame-by-frame VAE), view2 length = 4k+1 (chunk VAE)
        context_idx_orig = data["context_idx_orig"]
        inputs_shared = {
            "input_video": gt_video,
            "vace_video": vace_video,
            "vace_video_mask": data["vace_video_mask"],
            "vace_video_idx": data["vace_video_idx"],
            "context_idx_orig": context_idx_orig,
            "height": vace_video.shape[2],
            "width": vace_video.shape[3],
            "num_frames": gt_video.shape[0],
            "loss_latent_start": len(context_idx_orig),  # view2 latent starts after T_view1 (frame-by-frame → 1 latent per frame)
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 5,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        import time
        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        _sync(); t0 = time.perf_counter()
        if inputs is None: inputs = self.get_pipeline_inputs(data)   # Qwen captioner 포함
        _sync(); t_cap = time.perf_counter()
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        _sync(); t_xfer = time.perf_counter()

        # Pipeline units: ShapeChecker → TextEncoder → NoiseInit → InputVideoEmbedder(VAE) → VACE(VAE) → ...
        unit_times = {}
        for unit in self.pipe.units:
            _sync(); _t = time.perf_counter()
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
            _sync(); unit_times[type(unit).__name__] = time.perf_counter() - _t
        _sync(); t_units = time.perf_counter()

        loss = self.task_to_loss[self.task](self.pipe, *inputs)   # DiT forward + MSE
        _sync(); t_loss = time.perf_counter()

        print(
            f"[FWD] caption={t_cap - t0:.2f}s  xfer={t_xfer - t_cap:.2f}s  "
            f"units={t_units - t_xfer:.2f}s  dit_loss={t_loss - t_units:.2f}s  "
            f"| VAE={unit_times.get('WanVideoUnit_InputVideoEmbedder', 0):.2f}s  "
            f"VACE={unit_times.get('WanVideoUnit_VACE', 0):.2f}s  "
            f"TextEnc={unit_times.get('WanVideoUnit_PromptEmbedder', 0):.2f}s"
        )
        return loss

    @torch.no_grad()
    def predict_rendered_video(self, data):
        """gt_video와 모델 단일 스텝 x0 예측을 (T, H, W, 3) uint8로 반환."""
        pipe = self.pipe
        inputs = self.get_pipeline_inputs(data)
        prompt = inputs[1].get("prompt", "")  # inputs_posi
        inputs = self.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
        for unit in pipe.units:
            inputs = pipe.unit_runner(unit, pipe, *inputs)
        inputs_shared, inputs_posi, _ = inputs

        input_latents = inputs_shared.get("input_latents")  # GT latents, for gt decoding only

        # 50-step denoising (inference mode)
        num_steps = 50
        pipe.scheduler.set_timesteps(num_steps, training=False)
        # Initialize noise with the split-encoded latent shape (view1 frame-by-frame + view2 chunked)
        # so vace conditioning u and noise latent x have matching token counts.
        latents = torch.randn_like(input_latents)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

        for t in pipe.scheduler.timesteps:
            timestep = t.to(dtype=pipe.torch_dtype, device=pipe.device).unsqueeze(0)
            inputs_shared["latents"] = latents
            noise_pred = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            latents = pipe.scheduler.step(noise_pred, t, latents)

        # training mode로 복원
        pipe.scheduler.set_timesteps(1000, training=True)

        # VAE decode — view1 (keyframes) was encoded frame-by-frame, view2 (4k+1) as chunk
        # Split at T_lat_v1; trim view2 back to its original length (drop 4k+1 pad frames).
        context_idx_orig = inputs_shared.get("context_idx_orig")
        T_lat_v1 = len(context_idx_orig) if context_idx_orig is not None else latents.shape[2] // 2
        view2_len = int(len(data["target_idx_orig"]))   # original (un-padded) view2 frame count

        def decode_split(lats, view2_len=view2_len):
            # view1: decode each latent separately → 1 frame each
            d1 = torch.cat([
                pipe.vae.decode(lats[:, :, i:i+1], device=pipe.device)[0]
                for i in range(T_lat_v1)
            ], dim=1)  # (C, T_lat_v1, H, W)
            # view2: decode as chunk → (T_lat_v2 - 1)*4 + 1 frames, trim 4k+1 padding
            d2 = pipe.vae.decode(lats[:, :, T_lat_v1:], device=pipe.device)[0]
            d2 = d2[:, :view2_len]
            return torch.cat([d1, d2], dim=1)

        gt_all   = decode_split(input_latents) if input_latents is not None else None
        pred_all = decode_split(latents)

        def to_uint8(frames):
            frames = ((frames.float() + 1) * 127.5).clamp(0, 255)
            return frames.permute(1, 2, 3, 0).cpu().numpy().astype("uint8")  # (T, H, W, C)

        T_pred = pred_all.shape[1]  # T_lat_v1 + view2_len
        gt_frames   = to_uint8(gt_all[:, :T_pred]) if gt_all is not None else np.zeros((T_pred, pred_all.shape[2], pred_all.shape[3], 3), dtype="uint8")
        pred_frames = to_uint8(pred_all[:, :T_pred])  # (T, H, W, C) — view1 keyframes + view2 rendered

        def tensor_to_uint8(t):
            return (t.permute(0, 2, 3, 1).cpu().float().numpy() * 255).clip(0, 255).astype("uint8")

        vace_video = data["vace_video"]  # (T_view1 + T_view2, C, H, W), [0,1]
        gt_video   = data["gt_video"]    # (T_view1 + T_view2, C, H, W), [0,1]
        # view1/view2 boundary from mask (0→1 transition)
        _mask = data["vace_video_mask"]
        _ones = (_mask > 0.5).nonzero(as_tuple=True)[0]
        T_vace = int(_ones[0].item()) if len(_ones) > 0 else vace_video.shape[0] // 2

        # view2는 4k+1 padding 포함 → target_idx_orig 길이로 trim
        target_len = int(len(data["target_idx_orig"]))
        rendered_frames = tensor_to_uint8(vace_video[T_vace:T_vace + target_len])   # rendered (blurry) — view2
        context_frames  = tensor_to_uint8(vace_video[:T_vace])                       # context keyframes — view1
        gt_frames_view1 = tensor_to_uint8(gt_video[:T_vace])                         # view1 = first half
        gt_frames_view2 = tensor_to_uint8(gt_video[T_vace:T_vace + target_len])      # view2 = second half (padding 제외)

        return rendered_frames, gt_frames_view1, gt_frames_view2, pred_frames, context_frames, prompt


class WandbModelLogger(ModelLogger):
    def __init__(self, *args, wandb_project=None, wandb_name=None, log_video_steps=20, top_k_ckpts=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wandb = wandb_project is not None
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.log_video_steps = log_video_steps
        self.top_k_ckpts = top_k_ckpts
        self._ckpt_heap = []   # min-heap by psnr
        self._all_ckpts = []   # 저장된 모든 ckpt 경로
        self._last_ckpt_path = None
        self._wandb_initialized = False

    def _cleanup_ckpts(self):
        """top_k + last step만 남기고 나머지 삭제 (optim.pt 도 동반 삭제)."""
        top_k_paths = {path for _, _, path in self._ckpt_heap}
        keep = top_k_paths | ({self._last_ckpt_path} if self._last_ckpt_path else set())
        for path in self._all_ckpts:
            if path not in keep and os.path.exists(path):
                os.remove(path)
                optim_path = path.replace(".safetensors", ".optim.pt")
                if os.path.exists(optim_path):
                    os.remove(optim_path)

    def on_step_end(self, accelerator, model, save_steps=None, **kwargs):
        super().on_step_end(accelerator, model, save_steps, **kwargs)

        do_val = self.use_wandb and (self.num_steps % self.log_video_steps == 0)

        # save_model은 모든 rank가 함께 호출해야 함 (wait_for_everyone 때문)
        if do_val:
            ckpt_name = f"val-step-{self.num_steps}.safetensors"
            self.save_model(accelerator, model, ckpt_name)
            # optimizer/scheduler state 저장 → preempt 재개시 Adam moment 연속
            optimizer = kwargs.get("optimizer")
            scheduler = kwargs.get("scheduler")
            if accelerator.is_main_process and optimizer is not None:
                optim_path = os.path.join(
                    self.output_path, ckpt_name.replace(".safetensors", ".optim.pt")
                )
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "num_steps": self.num_steps,
                    },
                    optim_path,
                )
            if accelerator.is_main_process:
                self._all_ckpts.append(os.path.join(self.output_path, ckpt_name))

        if self.use_wandb and accelerator.is_main_process:
            if not self._wandb_initialized:
                wandb.init(project=self.wandb_project, name=self.wandb_name)
                self._wandb_initialized = True

            # 한 step당 wandb.log를 한 번만 호출 (step= 키워드로 x축 고정)
            log_dict = {}
            loss = kwargs.get("loss")
            if loss is not None:
                log_dict["loss"] = loss.item()

            if do_val:
                import numpy as np
                data = kwargs.get("data")
                if data is not None:
                    rendered_frames, gt_frames_view1, gt_frames_view2, pred_frames, context_frames, prompt = accelerator.unwrap_model(model).predict_rendered_video(data)

                    # 동적 길이: dataset이 알려준 원본 (un-padded) frame counts
                    view1_len = int(len(data["context_idx_orig"]))   # keyframe count
                    view2_len = int(len(data["target_idx_orig"]))    # rendered frame count (no pad)

                    # raw context/rendered/gt_view2는 4k+1 패딩 포함 → view2_len으로 잘라 패딩 제거
                    rendered_frames = rendered_frames[:view2_len]
                    gt_frames_view2 = gt_frames_view2[:view2_len]
                    context_frames  = context_frames[:view1_len]
                    gt_frames_view1 = gt_frames_view1[:view1_len]

                    # pred_frames 구조: 앞 view1_len = view1 pred, 뒤 view2_len = view2 pred
                    gt_frames_v       = gt_frames_view2
                    pred_frames_v     = pred_frames[view1_len:view1_len + view2_len]
                    rendered_frames_v = rendered_frames
                    context_frames_v  = context_frames

                    # PSNR / SSIM 계산 (gt vs pred, uint8 기준)
                    from skimage.metrics import structural_similarity as ssim_fn
                    mse_vals, ssim_vals = [], []
                    for g, p in zip(gt_frames_v, pred_frames_v):
                        gf, pf = g.astype(np.float32), p.astype(np.float32)
                        mse_vals.append(np.mean((gf - pf) ** 2))
                        ssim_vals.append(ssim_fn(g, p, channel_axis=-1, data_range=255))
                    mean_mse = np.mean(mse_vals)
                    psnr = 10 * np.log10(255.0 ** 2 / mean_mse) if mean_mse > 0 else 100.0
                    ssim = np.mean(ssim_vals)

                    ckpt_path = os.path.join(self.output_path, ckpt_name)

                    # last 업데이트
                    self._last_ckpt_path = ckpt_path

                    # top-k heap 관리
                    if len(self._ckpt_heap) < self.top_k_ckpts:
                        heapq.heappush(self._ckpt_heap, (psnr, self.num_steps, ckpt_path))
                    elif psnr > self._ckpt_heap[0][0]:
                        heapq.heapreplace(self._ckpt_heap, (psnr, self.num_steps, ckpt_path))

                    self._cleanup_ckpts()

                    # wandb 비디오 로깅: rendered | gt | pred 3단 비교 (view2 전체, 패딩 제거됨)
                    triple = np.stack([
                        np.concatenate([r, g, p], axis=1)
                        for r, g, p in zip(rendered_frames_v, gt_frames_v, pred_frames_v)
                    ])  # (view2_len, H, W*3, C)
                    video_tensor = triple.transpose(0, 3, 1, 2)
                    # val/view1: context | gt_view1 | pred_view1 (각 view1_len개)
                    pred_frames_view1 = pred_frames[:view1_len]
                    view1_video = np.stack([
                        np.concatenate([c, g, p], axis=1)
                        for c, g, p in zip(context_frames_v, gt_frames_view1, pred_frames_view1)
                    ]).transpose(0, 3, 1, 2)

                    pred_all_video = pred_frames.transpose(0, 3, 1, 2)  # (view1_len + view2_len, C, H, W)

                    scene_name = data.get("scene", "unknown")
                    # 비디오는 wandb.log → 워크스페이스 패널에 표시 (key 별 1 panel, step slider).
                    log_dict.update({
                        "val/psnr": psnr,
                        "val/ssim": ssim,
                        "val/rendered_gt_pred": wandb.Video(
                            video_tensor, fps=8, format="mp4",
                            caption=f"step {self.num_steps} | {scene_name} | rendered / gt / pred (view2, {view2_len} frames)",
                        ),
                        "val/view1": wandb.Video(
                            view1_video, fps=8, format="mp4",
                            caption=f"step {self.num_steps} | {scene_name} | context / gt_view1 / pred_view1 ({view1_len} keyframes)",
                        ),
                        "val/pred_all": wandb.Video(
                            pred_all_video, fps=8, format="mp4",
                            caption=f"step {self.num_steps} | {scene_name} | full pred (view1: 0~{view1_len-1}, view2: {view1_len}~{view1_len + view2_len - 1})",
                        ),
                    })
                    # 프롬프트는 wandb.run.summary 로 → 매 step 누적 안 됨 (최신 1개만 보존).
                    if wandb.run is not None:
                        wandb.run.summary["val/prompt"] = wandb.Html(
                            f"<p><b>{scene_name}</b><br>{prompt}</p>"
                        )

            if log_dict:
                log_dict["step"] = self.num_steps
                # step= 키워드를 넘기면 wandb 내부 x-axis 가 self.num_steps 로 고정되어
                # resume 시에도 loss/psnr/ssim 차트가 실제 step 부터 이어서 그려진다.
                wandb.log(log_dict, step=self.num_steps)


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument("--wandb_project", type=str, default='wan_vace')
    parser.add_argument("--wandb_name", type=str, default='wan_vace')
    parser.add_argument("--dataset", type=str, default="kubric",
                        choices=["kubric", "multicam", "mixed"],
                        help="Which deblur dataset to use (mixed = kubric + multicam).")
    return parser


def build_dataset(name: str):
    if name == "kubric":
        from diffsynth.core.data.dataset_kubric import DatasetSpringDeblur as Cls, DatasetSpringDeblurCfg as Cfg
        return Cls(Cfg())
    if name == "multicam":
        from diffsynth.core.data.dataset_multicam import DatasetMulticamDeblur as Cls, DatasetMulticamDeblurCfg as Cfg
        return Cls(Cfg())
    if name == "mixed":
        from diffsynth.core.data.mixed_dataset import build_kubric_multicam_vace
        return build_kubric_multicam_vace()
    raise ValueError(f"Unknown --dataset: {name}")


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    if args.wandb_name is not None:
        seoul_time = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M-%S")
        args.output_path = os.path.join(
            "./outputs",
            f"exp_{args.wandb_name}",
            seoul_time,
        )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = build_dataset(args.dataset)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = WandbModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
