from dataclasses import asdict
from types import SimpleNamespace
from typing import TypedDict

import lightning as L
import numpy as np
import torch

from utils.model_init import init_vae
from utils.training import set_kl_weight, set_lr
from utils.wandb_utils import load_pretrained_module_from_wandb


class LogVars(TypedDict):
    prog_bar: bool
    logger: bool
    on_step: bool
    on_epoch: bool
    sync_dist: bool


# Define the Lightning Module
class VAE_Lightning(L.LightningModule):
    def __init__(self, config, pretrained_wandb_config=None):
        super().__init__()

        # ------------------------------ training params ----------------------------- #
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.model = init_vae(config)

        # -------------------------- freeze / train denoiser ------------------------- #
        self.strict_loading = False

    def training_step(self, batch, batch_idx):
        total_loss, mse_loss, kl_loss = self.model.compute_loss(batch, self.current_kl_weight)

        log_vars: LogVars = {
            "prog_bar": True,
            "logger": True,
            "on_step": True,
            "on_epoch": False,
            "sync_dist": True,
        }

        self.log("train_loss", total_loss.item(), **log_vars)
        self.log("mse_loss", mse_loss.item(), **log_vars)
        self.log("kl_loss", kl_loss.item(), **log_vars)
        self.log("current_kl_weight", self.current_kl_weight, **log_vars)

        lrs = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lrs, **log_vars)

        return total_loss

    def configure_optimizers(self):
        params = self.model.parameters()

        optimizer = torch.optim.Adam(params, lr=self.config.lr_init)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: set_lr(self.config, epoch) / self.config.lr_init
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # Update the learning rate at every step
            "frequency": 1,  # Update every step
        }
        return [optimizer], [scheduler_config]

    def on_train_epoch_start(self):
        # Update the KL weight based on the current epoch
        self.current_kl_weight = set_kl_weight(self.config, self.current_epoch)

    def on_load_checkpoint(self, checkpoint):
        """Fix the checkpoint loading issue for deepspeed."""
        if "state_dict" in checkpoint:
            return
        state_dict = checkpoint["module"]
        state_dict = {k.partition("module.")[2]: state_dict[k] for k in state_dict.keys()}
        checkpoint["state_dict"] = state_dict
        return
