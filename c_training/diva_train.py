import sys
import time
if '/mnt/home/blyo1/hdiva' not in sys.path:
    sys.path.append('/mnt/home/blyo1/hdiva')
import os

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy

from a_datasets.celeba_lightning import CelebAColorDataModule
from c_training.diva_lightning import DiVA_Lightning
from c_training.configs.diva_config import DiVA_Manual_CelebA_64_Training_Config
from utils.training import get_checkpoint_dir, rename_checkpoint_folder, WandbArtifactCallback


# Training script
def main():
    # ---------------------------------- params ---------------------------------- #
    base_dir = "/mnt/home/blyo1/hdiva"
    config = DiVA_Manual_CelebA_64_Training_Config()

    # ------------------------------- run training ------------------------------- #
    # seed
    seed_everything(config.seed)

    # wandb
    wandb_logger = WandbLogger(
        project=config.model_name,
        # log_model="all",
        # log_model=True,
        log_model=False,
        save_dir=f"{base_dir}/c_training/lightning_logs",
        # checkpoint_name='{epoch:04d}'
        # checkpoint_name=f'{config.model_name}_{{epoch:04d}}'
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
        dirpath=f"{base_dir}/c_training/lightning_checkpoints/{config.model_name}",
        enable_version_counter=True,
        filename="{epoch:04d}",
    )

    wandb_callback = WandbArtifactCallback(every_n_epochs=config.checkpoint_every_n_epochs, config=config)

    # Initialize the dataset
    datamodule = CelebAColorDataModule(config)

    # Initialize the model
    if config.resume_from_checkpoint:
        if config.use_pretrained_denoiser_only:
            pretrained_wandb_config = {
                # 'project_name': 'ddpm_celeba_color_64',
                'project_name': 'diva_manual_celeba_color_64',
                'model_num': 84,
                'artifact_id': "v0",
                'checkpoint_dir': f"/mnt/home/blyo1/diva/celeba_color/lightning_checkpoints",
                # 'checkpoint_dir': config.model_checkpoint_dir,
            }
            model = DiVA_Lightning(config=config, pretrained_wandb_config=pretrained_wandb_config)
        else:
            pretrained_checkpoint_dir = get_checkpoint_dir(model_num=84, 
                                                           project_name=config.model_name, 
                                                           checkpoint_dir=config.model_checkpoint_dir, 
                                                           epoch="0299")
            model = DiVA_Lightning.load_from_checkpoint(pretrained_checkpoint_dir, config=config)
    else:
        model = DiVA_Lightning(config=config)

    # Initialize the Trainer
    trainer = Trainer(
        accelerator="cuda",
        devices=config.num_gpus_per_node,  # GPUs per node (adjust based on your setup)
        num_nodes=config.num_nodes,  # Number of nodes
        strategy=config.strategy,
        # strategy=DDPStrategy(),
        # strategy=FSDPStrategy(),
        max_epochs=config.num_epochs,
        logger=wandb_logger,
        log_every_n_steps=config.log_every_n_steps,
        # precision="16-mixed",  # Use mixed precision
        precision=config.precision,
        enable_checkpointing=True,
        enable_progress_bar=True,
        callbacks=[
            checkpoint_callback_total, 
            wandb_callback
        ],
        default_root_dir=f"{base_dir}",
    )

    # update the checkpoint callback's dirpath
    rename_checkpoint_folder(trainer, checkpoint_dir=os.path.join(base_dir, f"c_training/lightning_checkpoints/{config.model_name}"))
    
    # Log hyperparameters to wandb
    # wandb_logger.log_hyperparams(training_config)

    # Start training
    start_time = time.time()
    trainer.fit(model, datamodule=datamodule)
    end_time = time.time()
    time_taken = (end_time - start_time)/60

    # Log training time
    print(f"Training time: {time_taken} minutes")
    training_config = {}
    training_config["training_time_minutes"] = time_taken
    wandb_logger.log_hyperparams(training_config)

    # Finish wandb run
    wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()