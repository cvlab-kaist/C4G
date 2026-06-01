import os, torch, time
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    # Resume optimizer/scheduler/step state if checkpoint provided
    resumed_num_steps = 0
    if args is not None and getattr(args, "lora_checkpoint", None) is not None:
        optim_path = args.lora_checkpoint.replace(".safetensors", ".optim.pt")
        if os.path.exists(optim_path):
            state = torch.load(optim_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            resumed_num_steps = int(state.get("num_steps", 0))
            print(f"Optimizer/scheduler state restored from {optim_path} (num_steps={resumed_num_steps})")
        else:
            # Fallback: parse step number from filename like "val-step-460.safetensors"
            import re
            m = re.search(r"step-(\d+)\.safetensors$", args.lora_checkpoint)
            if m:
                resumed_num_steps = int(m.group(1))
                print(f"No optimizer state found at {optim_path}, starting fresh "
                      f"(num_steps={resumed_num_steps} parsed from ckpt filename).")
            else:
                print(f"No optimizer state found at {optim_path}, starting fresh (num_steps=0).")
    model_logger.num_steps = resumed_num_steps

    for epoch_id in range(num_epochs):
        _sync(); t_prev = time.perf_counter()
        for data in tqdm(dataloader):
            _sync(); t_data = time.perf_counter()
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                _sync(); t_fwd = time.perf_counter()
                accelerator.backward(loss)
                _sync(); t_bwd = time.perf_counter()
                optimizer.step()
                _sync(); t_opt = time.perf_counter()
                if torch.cuda.is_available():
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    print(
                        f"[STEP] dataload={t_data - t_prev:.2f}s  fwd={t_fwd - t_data:.2f}s  "
                        f"bwd={t_bwd - t_fwd:.2f}s  opt={t_opt - t_bwd:.2f}s  "
                        f"total={t_opt - t_prev:.2f}s  VRAM={peak_mb:.0f}MB ({peak_mb/1024:.1f}GB)"
                    )
                    torch.cuda.reset_peak_memory_stats()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss, data=data,
                                         optimizer=optimizer, scheduler=scheduler)
                scheduler.step()
                _sync(); t_prev = time.perf_counter()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
