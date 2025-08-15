import sys
from lightning.pytorch import Trainer, LightningModule, seed_everything
import lightning as L
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline
from models.scheduler import DDPMScheduler

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import math
import json
import wandb
from datasets import load_dataset
from autoclip.torch import QuantileClip

from models.ddpm import DDPM
from models.infnet import HalfUNetInfNet, LinearInfNet, HalfUNetInfNetNoTime
from celeba.utils.create_from_config import create_unet2d_model
from datasets.celeba import get_celeba_color_data
from dataclasses import asdict
import numpy as np


# Define the Lightning Module
class Lightning_Model(LightningModule):
    def __init__(self, config):
        super().__init__()

        # ------------------------------ training params ----------------------------- #
        self.config = config
        self.config_dict = asdict(config)  # turn the object attributes into a dictionary

        # ---------------------------- denoiser ----------------------------- #
        self.denoiser = create_unet2d_model(config)

        # --------------------------------- scheduler -------------------------------- #
        '''Load the DDPM pipeline from Huggingface and use its scheduler.'''
        scheduler_config = {}
        scheduler_config['num_train_timesteps'] = config.num_timesteps
        scheduler_config['prediction_type'] = "epsilon"  
        scheduler_config['beta_schedule'] = config.beta_schedule  # "linear" or "sigmoid" or "squaredcos_cap_v2"
        
        self.scheduler = DDPMScheduler(**scheduler_config)
        
        # set the num_inference_steps parameter (even though num_train_timesteps is given by new_config)
        self.scheduler.set_timesteps(num_inference_steps=config.num_timesteps)

        
        # --------------------------- define diva --------------------------- #
        self.ddpm = DDPM(self.denoiser, self.scheduler, config.kl_reduction, config.timestep_dist)

        # -------------------------- freeze / train denoiser ------------------------- #
        # Train the denoiser as well
        self.ddpm.denoiser.requires_grad = True
        for param in self.ddpm.denoiser.parameters():
            param.requires_grad = True
        
        self.save_hyperparameters()

        self.strict_loading = False

    def training_step(self, batch, batch_idx):
        # images = batch['images']  # Assuming batch contains images
        # Implement your loss function (replace with your actual loss logic)
        
        mse_loss = self.ddpm.compute_loss(batch)  # Define compute_loss in diva
        total_loss = mse_loss
        
        self.log("train_loss", total_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(self.ddpm.parameters(), lr=self.config.learning_rate_init)

        def lr_lambda(step):
            if step >= self.config.num_warmup_steps:
                return self.config.learning_rate_final / self.config.learning_rate_init
            progress = step / self.config.num_warmup_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            init_weight = cosine_factor
            final_weight = 1.0 - cosine_factor
            multiplier = (
                init_weight * 1.0 + 
                final_weight * (self.config.learning_rate_final / self.config.learning_rate_init)
            )
            return multiplier
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # Update the learning rate at every step
            "frequency": 1,  # Update every step
        }

        return [optimizer], [scheduler_config]

    # def on_train_epoch_start(self):
    #     # Update the KL weight based on the current epoch
    #     progress = min(1, self.current_epoch / (self.kl_annealing_epochs - 1))  # Normalize to [0, 1]
    #     if self.kl_annealing_schedule == "linear":
    #         self.current_kl_weight = self.kl_weight_min + (self.kl_weight_max - self.kl_weight_min) * progress
    #     elif self.kl_annealing_schedule == "cosine":
    #         cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
    #         current_kl_weight = self.kl_weight_min + (self.kl_weight_max - self.kl_weight_min) * (1-cosine_term)
    #         self.current_kl_weight = min(max(current_kl_weight, self.kl_weight_min), self.kl_weight_max)
    #     elif self.kl_annealing_schedule == "linear_in_log":
    #         self.current_kl_weight = self.kl_weight_min * (self.kl_weight_max / self.kl_weight_min) ** progress
    #     else:
    #         raise ValueError(f"Unknown kl_annealing_schedule: {self.kl_annealing_schedule}, expected 'linear' or 'cosine' or 'linear_in_log'")
        

    def train_dataloader(self):
        dataset, loader = get_celeba_color_data(with_label=False, batch_size=self.config.train_batch_size_per_gpu, dataset_size=self.config.dataset_size,)
        # batch size is gradient_accumulation_steps * train_micro_batch_size_per_gpu * num_gpus_per_node = 1 * 16 * 20 = 320
        return loader

    def on_save_checkpoint(self, checkpoint):
        if self.config.train_infnet_only:
            # remove the denoiser from the checkpoint
            checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if "denoiser" not in k}
            return
        else:
            return
        # # remove the denoiser weights from the checkpoint
        # checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if "denoiser" not in k}
        # return

    def on_load_checkpoint(self, checkpoint):
        """Fix the checkpoint loading issue for deepspeed."""
        if "state_dict" in checkpoint:
            return
        state_dict = checkpoint['module']
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        checkpoint['state_dict'] = state_dict
        return