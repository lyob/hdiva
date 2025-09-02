import sys
import torch
from dataclasses import asdict
import numpy as np

# import lightning
from lightning.pytorch import LightningModule
# from lightning.pytorch import Trainer, LightningModule, seed_everything
# import lightning as L
# from lightning.pytorch.strategies import DeepSpeedStrategy
# from lightning.pytorch.loggers import WandbLogger
# from lightning.pytorch.callbacks import ModelCheckpoint, Callback

# importing dataset
from a_datasets.dataset_utils import load_dataset

# import model
from utils.model_init import init_lvae

from utils.training import set_kl_weight

# Define the Lightning Module
class Lightning_Model(LightningModule):
    def __init__(self, config):
        super().__init__()

        # ------------------------------ training params ----------------------------- #
        self.config = config
        # self.config_dict = asdict(config)  # turn the object attributes into a dictionary
        self.model = init_lvae(config)
        self.criterion = self.model.criterion
        self.save_hyperparameters(asdict(config))
        self.strict_loading = False
        self.current_kl_weights = np.zeros((2))

    def training_step(self, batch, batch_idx):
        '''Calculates the loss at every step, for a given criterion'''

        mse_loss, kl_losses = self.criterion(batch, self.config.input_dim)
        # print("current_kl_weights:", self.current_kl_weights)

        # kl_losses is a list of tensors. multiply each item in the list with the corresponding element in the tensor self.current_kl_weights and sum the result
        # weighted_kl_losses = torch.dot(kl_losses, self.current_kl_weights)

        weighted_kl = []
        for loss, weight in zip(kl_losses, self.current_kl_weights):
            weighted_kl.append(loss * weight)  # numpy scalar is auto-converted
        weighted_kl = sum(weighted_kl)

        total_loss = mse_loss + weighted_kl

        # logging every training step
        self.log("mse_loss", mse_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for i, kl_loss in enumerate(kl_losses):
            self.log(f"kl_loss_z{i+1}", kl_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"kl_loss_total", weighted_kl.sum().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_loss", total_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return total_loss

    def configure_optimizers(self):
        '''Define optimizer'''
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate_init)
        return optimizer

    def on_train_epoch_start(self):
        '''Update the KL weight based on the current epoch'''
        self.current_kl_weights = set_kl_weight(self.config, self.current_epoch)

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