import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import math
import json
import wandb

# import lightning
from lightning.pytorch import Trainer, LightningModule, seed_everything
import lightning as L
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

# importing dataset
from a_datasets.hdisks3 import load_dataset
from dataclasses import asdict

# import model
from utils.model_init import init_vae

from utils.training import set_kl_weight

# Define the Lightning Module
class Lightning_Model(LightningModule):
    def __init__(self, config):
        super().__init__()

        # ------------------------------ training params ----------------------------- #
        self.config = config
        # self.config_dict = asdict(config)  # turn the object attributes into a dictionary
        self.model = init_vae(config)
        self.criterion = self.model.criterion
        self.save_hyperparameters(asdict(config))
        self.strict_loading = False
        self.current_kl_weight = 0

    def training_step(self, batch, batch_idx):
        '''Calculates the loss at every step, for a given criterion'''

        mse_loss, kl_loss = self.criterion(batch)
        total_loss = mse_loss + self.current_kl_weight * kl_loss

        # logging every training step
        self.log("mse_loss", mse_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("kl_loss", kl_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_loss", total_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return total_loss

    def configure_optimizers(self):
        '''Define optimizer'''
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate_init)
        return optimizer

    def on_train_epoch_start(self):
        '''Update the KL weight based on the current epoch'''
        self.current_kl_weight = set_kl_weight(self.config, self.current_epoch)

    def train_dataloader(self):
        '''get the dataloader for the training dataset'''
        dataloader = load_dataset(self.config)
        # batch size is gradient_accumulation_steps * train_micro_batch_size_per_gpu * num_gpus_per_node = 1 * 16 * 20 = 320
        return dataloader

    def on_load_checkpoint(self, checkpoint):
        """Fix the checkpoint loading issue for deepspeed."""
        if "state_dict" in checkpoint:
            return
        state_dict = checkpoint['module']
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        checkpoint['state_dict'] = state_dict
        return