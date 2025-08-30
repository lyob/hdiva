import sys
from lightning.pytorch import Trainer, LightningModule, seed_everything
import lightning as L
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from celeba_color.models.diva import DiVA
from celebahq.models.infnet import HalfUNetInfNet, LinearInfNet, HalfUNetInfNetNoTime
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
from celeba.utils.create_from_config import create_unet2d_model
from celeba.utils.dataset import get_celeba_color_data
from dataclasses import asdict


# Define the Lightning Module
class DiVA_Lightning(LightningModule):
    def __init__(self, config):
        super().__init__()

        # ------------------------------ training params ----------------------------- #
        self.config = config
        self.config_dict = asdict(config)  # turn the object attributes into a dictionary

        # ---------------------------- denoiser ----------------------------- #
        self.denoiser = create_unet2d_model(config)

        # --------------------------------- scheduler -------------------------------- #
        '''Load the DDPM pipeline from Huggingface and use its scheduler.'''
        # ddpm_id = "google/ddpm-celebahq-256"
        # pipe = DDPMPipeline.from_pretrained(ddpm_id, cache_dir="/mnt/home/blyo1/ceph/huggingface_cache/")
        # scheduler_config = pipe.scheduler.config.copy()

        scheduler_config = {}
        scheduler_config['num_train_timesteps'] = config.num_timesteps
        scheduler_config['prediction_type'] = "epsilon"  
        scheduler_config['beta_schedule'] = config.beta_schedule  # "linear" or "sigmoid" or "squaredcos_cap_v2"
        self.scheduler = DDPMScheduler(**scheduler_config)

        # set the num_inference_steps parameter (even though num_train_timesteps is given by new_config)
        self.scheduler.set_timesteps(num_inference_steps=config.num_timesteps)

        # ----------------------------- inference network ---------------------------- #
        if config.infnet_type == "half_unet":
            infnet_config = self.denoiser.config

            # smaller network
            infnet_config["block_out_channels"] = config.infnet_block_out_channels

            # default activation function is "silu"
            infnet_config["act_fn"] = config.infnet_act_fn if config.infnet_act_fn is not None else "silu"

            infnet = HalfUNetInfNet(**infnet_config, dim_latent=config.d_latent)
        elif config.infnet_type == "half_unet_no_time_no_attn":
            infnet_config = self.denoiser.config

            # smaller network
            infnet_config["block_out_channels"] = config.infnet_block_out_channels

            # default activation function is "relu"
            infnet_config["act_fn"] = config.infnet_act_fn if config.infnet_act_fn is not None else "relu"

            # remove the attention blocks
            if len(config.infnet_block_out_channels) == 6:
                infnet_config["down_block_types"] = ['DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'DownBlock2D',]
            elif len(config.infnet_block_out_channels) == 4:
                infnet_config["down_block_types"] = ['DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'DownBlock2D']
            
            infnet_config["add_attention"] = False

            # remove the time embedding
            infnet = HalfUNetInfNetNoTime(**infnet_config, dim_latent=config.d_latent)
        else:
            raise ValueError(f"Unknown infnet_type: {config.infnet_type}, expected 'linear' or 'half_unet' or 'half_unet_no_time'")

        # --------------------------- define diva --------------------------- #
        self.diva = DiVA(self.denoiser, infnet, self.scheduler, config.kl_reduction, config.timestep_dist)

        # -------------------------- freeze / train denoiser ------------------------- #
        if config.train_infnet_only:
            # Freeze the weights of the denoiser
            self.diva.denoiser.requires_grad = False
            for param in self.diva.denoiser.parameters():
                param.requires_grad = False
        else:
            # Train the denoiser as well
            self.diva.denoiser.requires_grad = True
            for param in self.diva.denoiser.parameters():
                param.requires_grad = True
        
        # ----------------------------- kl weight params ----------------------------- #
        self.kl_weight_min = config.kl_weight_min
        self.kl_weight_max = config.kl_weight_max
        self.current_kl_weight = config.kl_weight_min
        self.kl_annealing_epochs = config.kl_annealing_epochs
        self.kl_annealing_schedule = config.kl_annealing_schedule

        self.save_hyperparameters()

        self.strict_loading = False


    def training_step(self, batch, batch_idx):
        # images = batch['images']  # Assuming batch contains images
        # Implement your loss function (replace with your actual loss logic)
        
        if self.config.infnet_type == "linear" or self.config.infnet_type == "half_unet_no_time_no_attn":
            mse_loss, kl_loss = self.diva.compute_loss(batch)  # Define compute_loss in diva
            total_loss = mse_loss + self.current_kl_weight * kl_loss
        elif self.config.infnet_type == "half_unet" or self.config.infnet_type == "half_unet_no_time":
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                mse_loss, kl_loss = self.diva.compute_loss(batch)  # Define compute_loss in diva
                total_loss = mse_loss + self.current_kl_weight * kl_loss
        else:
            raise ValueError(f"Unknown infnet_type: {self.infnet_type}, expected 'linear' or 'half_unet'")
        
        self.log("train_loss", total_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("mse_loss", mse_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("kl_loss", kl_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("current_kl_weight", self.current_kl_weight, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(self.diva.parameters(), lr=self.config.learning_rate_init)

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

    def on_train_epoch_start(self):
        # Update the KL weight based on the current epoch
        progress = min(1, self.current_epoch / (self.kl_annealing_epochs - 1))  # Normalize to [0, 1]
        if self.kl_annealing_schedule == "linear":
            self.current_kl_weight = self.kl_weight_min + (self.kl_weight_max - self.kl_weight_min) * progress
        elif self.kl_annealing_schedule == "cosine":
            cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
            current_kl_weight = self.kl_weight_min + (self.kl_weight_max - self.kl_weight_min) * (1-cosine_term)
            self.current_kl_weight = min(max(current_kl_weight, self.kl_weight_min), self.kl_weight_max)
        elif self.kl_annealing_schedule == "linear_in_log":
            self.current_kl_weight = self.kl_weight_min * (self.kl_weight_max / self.kl_weight_min) ** progress
        else:
            raise ValueError(f"Unknown kl_annealing_schedule: {self.kl_annealing_schedule}, expected 'linear' or 'cosine' or 'linear_in_log'")
        

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