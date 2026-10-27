import os
from dataclasses import fields
from pathlib import Path

import hydra
import torch
import wandb
import signal
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from src.misc.weight_modify import checkpoint_filter_fn
from src.model.load_foundation_model import load_foundation_model

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper



def apply_global_dataset_options(cfg) -> None:
    if not cfg.force_davis_intrinsics:
        return

    for dataset_wrapper in cfg.dataset:
        (field,) = fields(type(dataset_wrapper))
        dataset_cfg = getattr(dataset_wrapper, field.name)
        dataset_cfg.force_davis_intrinsics = True


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def _adapt_token_tensor(
    tensor: torch.Tensor,
    target_count: int,
    ablation: str,
    noise_std: float,
    noise_seed: int,
    add_noise: bool,
) -> torch.Tensor:
    source_count = tensor.shape[0]
    if source_count == target_count:
        return tensor

    if ablation == "first1024":
        if target_count > source_count:
            raise ValueError(
                f"Cannot slice {source_count} pretrained gaussian tokens to {target_count}."
            )
        return tensor[:target_count].clone()

    if ablation == "duplicate_noise":
        if target_count != source_count * 2:
            raise ValueError(
                "duplicate_noise expects target gaussian token count to be exactly "
                f"2x the pretrained count, got {source_count} -> {target_count}."
            )
        duplicate = tensor.clone()
        if add_noise and noise_std > 0:
            generator = torch.Generator()
            generator.manual_seed(noise_seed)
            noise = torch.randn(
                duplicate.shape,
                generator=generator,
                dtype=duplicate.dtype,
                device=duplicate.device,
            )
            duplicate = duplicate + noise_std * noise
        return torch.cat([tensor, duplicate], dim=0)

    raise ValueError(f"Unknown gaussian_token_ablation: {ablation}")


def adapt_pretrained_gaussian_tokens(state_dict: dict, encoder) -> dict:
    ablation = getattr(encoder.cfg, "gaussian_token_ablation", "none")
    if ablation == "none":
        return state_dict

    token_key = "gaussian_tokens"
    if token_key not in state_dict:
        return state_dict

    target_count = encoder.gaussian_tokens.shape[0]
    source_count = state_dict[token_key].shape[0]
    if source_count == target_count:
        return state_dict

    state_dict = dict(state_dict)
    noise_std = getattr(encoder.cfg, "gaussian_token_noise_std", 1e-4)
    noise_seed = getattr(encoder.cfg, "gaussian_token_noise_seed", 1234)

    state_dict[token_key] = _adapt_token_tensor(
        state_dict[token_key],
        target_count,
        ablation,
        noise_std,
        noise_seed,
        add_noise=True,
    )

    anchor_key = "anchor_positions"
    if anchor_key in state_dict:
        state_dict[anchor_key] = _adapt_token_tensor(
            state_dict[anchor_key],
            target_count,
            ablation,
            noise_std,
            noise_seed,
            add_noise=False,
        )

    print(cyan(
        "Applied gaussian token ablation "
        f"'{ablation}': pretrained {source_count} -> model {target_count} tokens."
    ))
    return state_dict


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    apply_global_dataset_options(cfg)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            save_last=cfg.checkpointing.save_last,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    cuda_device_count = torch.cuda.device_count()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices=cuda_device_count if cuda_device_count > 0 else "auto",
        strategy=(
            DDPStrategy(find_unused_parameters=True,
                        broadcast_buffers=False,
                        gradient_as_bucket_view=True,
                        cluster_environment=LightningEnvironment())
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    
    vggt = load_foundation_model(cfg)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    if cfg.model.encoder.pretrained_weights:
        weight_path = cfg.model.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location='cpu')
        if 'model' in ckpt_weights:
            ckpt_weights = ckpt_weights['model']
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
            ckpt_weights = adapt_pretrained_gaussian_tokens(ckpt_weights, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        elif 'state_dict' in ckpt_weights:
            ckpt_weights = ckpt_weights['state_dict']
            ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
            ckpt_weights = adapt_pretrained_gaussian_tokens(ckpt_weights, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        elif isinstance(ckpt_weights, dict):
            new_ckpt = {}
            for key, value in ckpt_weights.items():
                if 'aggregator' in key:
                    new_ckpt[f'backbone.{key}'] = value
                if 'point_head' in key:
                    new_ckpt[key.replace('point_head', 'dpt_head')] = value
            new_ckpt = adapt_pretrained_gaussian_tokens(new_ckpt, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(new_ckpt, strict=False)
            del new_ckpt
        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")
        
        del ckpt_weights

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker,
        vggt=vggt,
        mode=cfg.mode,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
    torch.cuda.empty_cache()

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()
