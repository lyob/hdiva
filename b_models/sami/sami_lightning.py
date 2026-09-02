from dataclasses import asdict
from types import SimpleNamespace
from typing import TypedDict

import lightning as L
import numpy as np
import torch
import math

from utils.model_init import init_sami_model
from utils.wandb_utils import load_pretrained_module_from_wandb

# Define the Lightning Module
class SAMI_Lightning(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        # ------------------------------ training params ----------------------------- #
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.model = init_sami_model(config)

        # -------------------------- freeze / train denoiser ------------------------- #
        if config.use_pretrained_denoiser_only:
            # Load pretrained denoiser weights from wandb if not resuming from checkpoint
            wandb_config = dict(
                project_name = self.config.pretrained_model_name,
                model_num = self.config.pretrained_model_number,
                artifact_id = self.config.pretrained_artifact_id,
                checkpoint_dir = self.config.model_checkpoint_dir
            )
            self.model = load_pretrained_module_from_wandb(
                wandb_config, new_model=self.model, module_name="denoiser"
            )

            if config.train_infnet_only:
                # Freeze the weights of the denoiser
                self.model.denoiser.requires_grad = False

                for param in self.model.denoiser.parameters():
                    param.requires_grad = False
            else:
                self.model.denoiser.requires_grad = True
                for param in self.model.parameters():
                    param.requires_grad = True

        self.strict_loading = False

    def training_step(self, batch, batch_idx):
        # mse_loss = self.model.compute_loss(batch)
        total_loss, mse_loss, marginal_rate_loss = self.model.compute_loss(batch, beta=self.beta)

        log_vars: LogVars = {
            "prog_bar": True,
            "logger": True,
            "on_step": True,
            "on_epoch": False,
            "sync_dist": True,
        }

        self.log("train_loss", total_loss.item(), **log_vars)
        self.log("mse_loss", mse_loss.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("instantaneous rate", marginal_rate_loss.item(), **log_vars)
        if batch_idx == 0:
            # we also want to measure how much the infnet is learning via the rate
            cumulative_rate_t0, cumulative_rate_0t, kl_0 = self.model.compute_rate(batch)
            
            self.log("cumulative rate t-0", cumulative_rate_t0.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("cumulative rate 0-t", cumulative_rate_0t.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("kl(q(z|x_0)||N(0,I))", kl_0.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("beta", self.beta, **log_vars)
        
        lrs = [pg["lr"] for pg in self.optimizers().param_groups]
        self.log("learning_rate", lrs[0], **log_vars)

        return total_loss

    def configure_optimizers(self):
        # Define optimizer
        lr_init = self.config.lr_init
        lr_final = self.config.lr_final
        lr_num_warmup_epochs = self.config.lr_num_warmup_epochs
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_init)
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr_init)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        lr_schedule = self.config.lr_schedule
        lr_lambda = lambda epoch: set_lr(lr_init, lr_final, lr_num_warmup_epochs, lr_schedule, epoch) / lr_init
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # Update the learning rate at every step
            "frequency": 1,  # Update every step
        }
        return [optimizer], [scheduler_config]

    def on_save_checkpoint(self, checkpoint):
        if self.config.train_infnet_only:
            # remove the denoiser from the checkpoint
            checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if "denoiser" not in k}
            return
        else:
            return

    def on_train_epoch_start(self):
        self.beta = set_beta(self.config, self.current_epoch)

    def on_load_checkpoint(self, checkpoint):
        """Fix the checkpoint loading issue for deepspeed."""
        if "state_dict" in checkpoint:
            return
        state_dict = checkpoint["module"]
        state_dict = {k.partition("module.")[2]: state_dict[k] for k in state_dict.keys()}
        checkpoint["state_dict"] = state_dict
        return



class LogVars(TypedDict):
    prog_bar: bool
    logger: bool
    on_step: bool
    on_epoch: bool
    sync_dist: bool


def set_lr(lr_init, lr_final, lr_num_warmup_epochs, lr_schedule, current_epoch):
    if current_epoch >= lr_num_warmup_epochs:
        return lr_final

    if lr_schedule == "constant":
        lr = lr_init
    elif lr_schedule == "linear":
        lr = lr_init + (lr_final - lr_init) * (current_epoch / lr_num_warmup_epochs)
    elif lr_schedule == "cosine":
        lr = lr_final + 0.5 * (lr_init - lr_final) * (
            1 + math.cos(math.pi * current_epoch / lr_num_warmup_epochs)
        )
    elif lr_schedule == "exponential" or lr_schedule == "linear_in_log":
        lr = lr_init * (lr_final / lr_init) ** (current_epoch / lr_num_warmup_epochs)
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")
    return lr

def set_beta(config, current_epoch):
    beta_annealing_schedule = config.beta_annealing_schedule
    beta_wait_epochs = config.beta_wait_epochs
    beta_annealing_epochs = config.beta_annealing_epochs

    beta_init = config.beta_init
    beta_final = config.beta_final

    if current_epoch < beta_wait_epochs:
        return beta_init
    progress = min(1, (current_epoch-beta_wait_epochs) / (beta_annealing_epochs - 1))  # Normalize to [0, 1]

    if beta_annealing_schedule == "constant":
        current_beta = beta_final
    elif beta_annealing_schedule == "linear":
        current_beta = beta_init + (beta_final - beta_init) * progress
    elif beta_annealing_schedule == "cosine":
        cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
        current_beta = beta_init + (beta_final - beta_init) * (1 - cosine_term)
        current_beta = min(max(current_beta, beta_init), beta_final)
    elif beta_annealing_schedule == "exponential":
        current_beta = beta_init * (beta_final / beta_init) ** progress
    else:
        raise ValueError(
            f"Unknown beta_annealing_schedule: {beta_annealing_schedule}, expected 'constant' or 'linear' or 'cosine' or 'exponential'"
        )
    return current_beta

