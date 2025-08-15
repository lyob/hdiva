import sys
if '/mnt/home/blyo1/hdiva' not in sys.path:
    sys.path.append('/mnt/home/blyo1/hdiva')

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import os
from models.lightning import Lightning_Model
from models.config import DiVA_CelebA_64_Training_Config
from utils.training import rename_checkpoint_folder


# Training script
def main():
    # ---------------------------------- params ---------------------------------- #
    base_dir = "/mnt/home/blyo1/diva"
    config = DiVA_CelebA_64_Training_Config()

    # ------------------------------- run training ------------------------------- #
    # seed
    seed_everything(config.seed)

    # wandb
    wandb_logger = WandbLogger(
        project=config.model_name,
        log_model="all",
        # log_model=True,
        save_dir=f"{base_dir}/celeba_color/lightning_logs",
        # checkpoint_name=f'{model_name}-{wandb.run.id}'
    )
    
    # checkpointing
    checkpoint_callback_total = ModelCheckpoint(
        every_n_epochs=config.checkpoint_every_n_epochs,
        save_top_k=3,
        monitor="mse_loss",
        mode="min",
        save_weights_only=True,
        save_last=True,
        save_on_train_epoch_end=True,
        dirpath=f"{base_dir}/celeba_color/lightning_checkpoints/tmp",
        enable_version_counter=True,
        filename="{epoch:04d}",
    )

    # Initialize the model
    model = Lightning_Model(config=config)

    # Initialize the Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=config.num_gpus_per_node,  # GPUs per node (adjust based on your setup)
        num_nodes=config.num_nodes,  # Number of nodes
        strategy=config.strategy,
        max_epochs=config.num_epochs,
        logger=wandb_logger,
        log_every_n_steps=config.log_every_n_steps,
        # precision="16-mixed",  # Use mixed precision
        precision=config.precision,
        enable_checkpointing=True,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback_total],
        default_root_dir=f"{base_dir}",
    )

    # update the checkpoint callback's dirpath
    rename_checkpoint_folder(trainer, checkpoint_dir=os.path.join(base_dir, "celeba_color/lightning_checkpoints"))
    
    # Log hyperparameters to wandb
    training_config = {
        "seed" : config.seed,
        "strategy" : config.strategy,
        "num_epochs" : config.num_epochs,
        "batch_size_per_gpu" : config.train_batch_size_per_gpu,
        "precision" : config.precision,
    }
    wandb_logger.log_hyperparams(training_config)

    # Start training
    trainer.fit(model)

    # Finish wandb run
    wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()