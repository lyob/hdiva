import os
import numpy as np
import json
import math
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def rename_checkpoint_folder(
        trainer: Trainer, 
        checkpoint_dir: str,
    ):
    # Check if the current process is the global rank 0
    if trainer.global_rank == 0:
        # Get the WandbLogger from the trainer
        wandb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break
        
        if wandb_logger is None:
            print("No WandbLogger found. Skipping rename.")
            return


        # Find the ModelCheckpoint callback
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback is None:
            print("No ModelCheckpoint callback found. Skipping rename.")
            return

        # Determine the new folder name based on wandb experiment id or name
        # experiment name is something like "word-word-25"
        # get the number part
        experiment_number = wandb_logger.experiment.name.split("-")[-1]
        new_folder_name = f"{experiment_number}-{wandb_logger.experiment.name}-{wandb_logger.experiment.id}"
        new_folder = os.path.join(checkpoint_dir, new_folder_name)

        # create tmp folder
        tmp_folder = os.path.join(checkpoint_dir, 'tmp')
        os.makedirs(tmp_folder, exist_ok=True)
        os.rename(tmp_folder, new_folder)
        print(f"Renamed checkpoint folder from {tmp_folder} to {new_folder}")
        
        # Update the checkpoint callback's dirpath
        checkpoint_callback.dirpath = new_folder
        print(f"Updated checkpoint callback dirpath to {new_folder}")

def set_lr(config, current_epoch):
    lr_init = config.lr_init
    lr_final = config.lr_final
    if current_epoch >= config.lr_num_warmup_epochs:
        return config.lr_final

    if config.lr_schedule == "constant":
        lr = lr_init
    elif config.lr_schedule == "linear":
        lr = lr_init + (lr_final - lr_init) * (current_epoch / config.lr_num_warmup_epochs)
    elif config.lr_schedule == "cosine":
        lr = lr_final + 0.5 * (lr_init - lr_final) * (1 + math.cos(math.pi * current_epoch / config.lr_num_warmup_epochs))
    elif config.lr_schedule == "exponential":
        lr = lr_init * (lr_final / lr_init) ** (current_epoch / config.lr_num_warmup_epochs)
    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")
    return lr

def set_kl_weight(config, current_epoch):
    kl_annealing_schedule = config.kl_annealing_schedule
    kl_annealing_epochs = config.kl_annealing_epochs
    kl_weight_min = config.kl_weight_min if hasattr(config, 'kl_weight_min') else np.array(config.kl_weights_min)
    kl_weight_max = config.kl_weight_max if hasattr(config, 'kl_weight_max') else np.array(config.kl_weights_max)

    progress = min(1, current_epoch / (kl_annealing_epochs - 1))  # Normalize to [0, 1]

    if kl_annealing_schedule == "constant":
        current_kl_weight = kl_weight_max
    elif kl_annealing_schedule == "linear":
        current_kl_weight = kl_weight_min + (kl_weight_max - kl_weight_min) * progress
    elif kl_annealing_schedule == "cosine":
        cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
        current_kl_weight = kl_weight_min + (kl_weight_max - kl_weight_min) * (1-cosine_term)
        current_kl_weight = min(max(current_kl_weight, kl_weight_min), kl_weight_max)
    elif kl_annealing_schedule == "exponential":
        current_kl_weight = kl_weight_min * (kl_weight_max / kl_weight_min) ** progress
    else:
        raise ValueError(f"Unknown kl_annealing_schedule: {kl_annealing_schedule}, expected 'constant' or 'linear' or 'cosine' or 'exponential'")
    return current_kl_weight
