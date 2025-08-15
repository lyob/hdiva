import os
import json
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