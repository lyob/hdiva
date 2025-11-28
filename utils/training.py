import math
import os

import numpy as np
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


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
        tmp_folder = os.path.join(checkpoint_dir, "tmp")
        os.makedirs(tmp_folder, exist_ok=True)
        os.rename(tmp_folder, new_folder)
        print(f"Renamed checkpoint folder from {tmp_folder} to {new_folder}")

        # Update the checkpoint callback's dirpath
        checkpoint_callback.dirpath = new_folder
        print(f"Updated checkpoint callback dirpath to {new_folder}")


def get_checkpoint_dir(
    model_num=61,
    project_name: str = "diva_manual_celeba_color_64",
    checkpoint_dir: str = "/mnt/home/blyo1/hdiva/c_training/lightning_checkpoints",
    epoch: str | None = None,
):
    """find the checkpoint starting with a two digit number"""
    if isinstance(model_num, int):
        model_num = str(model_num)

    checkpoint_dir = f"{checkpoint_dir}/{project_name}"
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist.")

    for dir in os.listdir(checkpoint_dir):
        if dir.startswith(model_num):
            checkpoint_dir = f"{checkpoint_dir}/{dir}"
            break

    if epoch is not None and epoch != "last":
        checkpoint_dir = f"{checkpoint_dir}/epoch={epoch}.ckpt"
    else:
        checkpoint_dir = f"{checkpoint_dir}/last.ckpt"
    return checkpoint_dir


def set_lr(config, current_epoch):
    lr_init = config.lr_init
    lr_final = config.lr_final
    if current_epoch >= config.lr_num_warmup_epochs:
        return config.lr_final

    if config.lr_schedule == "constant":
        lr = lr_init
    elif config.lr_schedule == "linear":
        lr = lr_init + (lr_final - lr_init) * (
            current_epoch / config.lr_num_warmup_epochs
        )
    elif config.lr_schedule == "cosine":
        lr = lr_final + 0.5 * (lr_init - lr_final) * (
            1 + math.cos(math.pi * current_epoch / config.lr_num_warmup_epochs)
        )
    elif config.lr_schedule == "exponential":
        lr = lr_init * (lr_final / lr_init) ** (
            current_epoch / config.lr_num_warmup_epochs
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")
    return lr


def set_kl_weight(config, current_epoch):
    kl_annealing_schedule = config.kl_annealing_schedule
    kl_annealing_epochs = config.kl_annealing_epochs
    kl_weight_min = (
        config.kl_weight_min
        if hasattr(config, "kl_weight_min")
        else np.array(config.kl_weights_min)
    )
    kl_weight_max = (
        config.kl_weight_max
        if hasattr(config, "kl_weight_max")
        else np.array(config.kl_weights_max)
    )

    progress = min(1, current_epoch / (kl_annealing_epochs - 1))  # Normalize to [0, 1]

    if kl_annealing_schedule == "constant":
        current_kl_weight = kl_weight_max
    elif kl_annealing_schedule == "linear":
        current_kl_weight = kl_weight_min + (kl_weight_max - kl_weight_min) * progress
    elif kl_annealing_schedule == "cosine":
        cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
        current_kl_weight = kl_weight_min + (kl_weight_max - kl_weight_min) * (
            1 - cosine_term
        )
        current_kl_weight = min(max(current_kl_weight, kl_weight_min), kl_weight_max)
    elif kl_annealing_schedule == "exponential":
        current_kl_weight = kl_weight_min * (kl_weight_max / kl_weight_min) ** progress
    else:
        raise ValueError(
            f"Unknown kl_annealing_schedule: {kl_annealing_schedule}, expected 'constant' or 'linear' or 'cosine' or 'exponential'"
        )
    return current_kl_weight


# class WandbArtifactCallback(Callback):
#     def __init__(self, every_n_epochs=10):
#         self.every_n_epochs = every_n_epochs

#     def on_train_epoch_end(self, trainer, pl_module):
#         # Only log artifact every N epochs
#         if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
#             # Get the checkpoint path
#             ckpt_path = trainer.checkpoint_callback.best_model_path
#             if ckpt_path:
#                 # Log as artifact
#                 artifact = wandb.Artifact(
#                     name=f"model-{wandb.run.id}",
#                     type="model",
#                     metadata={"epoch": trainer.current_epoch}
#                 )
#                 artifact.add_file(ckpt_path)
#                 wandb.log_artifact(artifact)


class WandbArtifactCallback(Callback):
    def __init__(self, every_n_epochs: int, config):
        super().__init__()  # Add this line - it's crucial!
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, lightning_module):
        # Only log artifact every N epochs
        # if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
        if (
            trainer.is_global_zero
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        ):
            # Get the most recent checkpoint path
            ckpt_path = trainer.checkpoint_callback.last_model_path

            if ckpt_path and os.path.exists(ckpt_path):
                if trainer.global_rank == 0:
                    # Get the WandbLogger from the trainer
                    wandb_logger = None
                    for logger in trainer.loggers:
                        if isinstance(logger, WandbLogger):
                            wandb_logger = logger
                            break

                # Log as artifact
                artifact = wandb.Artifact(
                    name=f"model-{wandb_logger.experiment.id}",
                    type="model",
                    metadata={
                        "epoch": trainer.current_epoch,
                        "global_step": trainer.global_step,
                    },
                )
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)
                # print(f"Logged checkpoint artifact at epoch {trainer.current_epoch}")
