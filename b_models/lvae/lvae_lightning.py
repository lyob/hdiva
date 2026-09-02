import sys
import torch
from dataclasses import asdict
import numpy as np

# import lightning
from lightning.pytorch import LightningModule

# importing dataset
from a_datasets.dataset_utils import load_dataset, create_dataloader

# import model and training utils
from utils.model_init import init_lvae
from utils.training import set_kl_weight, set_lr


# Define the Lightning Module
class Lightning_Model(LightningModule):
    def __init__(self, config):
        super().__init__()

        # ------------------------------ training params ----------------------------- #
        self.config = config
        self.save_hyperparameters(asdict(config))

        # set up model
        self.model = init_lvae(config)
        self.strict_loading = False
        self.current_kl_weights = np.zeros((2))
        self.train_data = None

    def setup(self, stage: str):
        """
        Setup function called at the beginning of fit and test
        """
        if stage == 'fit':
            # we can set up anything required for training here
            print("Loading and moving dataset to GPU...")
            # 1. Generate the entire dataset on the CPU
            data_tensor_cpu = load_dataset(self.config)
            
            # 2. Move the entire tensor to the correct device (GPU) and store it
            self.train_data = data_tensor_cpu.to(self.device)
            print(f"Dataset moved to {self.train_data.device}")

    def train_dataloader(self):
        """
        Create a DataLoader that samples from the pre-loaded GPU tensor.
        """
        dataloader = create_dataloader(self.config, self.train_data)
        return dataloader

    def training_step(self, batch, batch_idx):
        '''Calculates the loss at every step, for a given criterion'''
        total_loss, mse_loss, weighted_kl = self.model.criterion(batch, self.current_kl_weights)

        log_vars = dict(
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # logging every training step
        self.log("mse_loss", mse_loss.item(), **log_vars)
        for i, kl_loss in enumerate(self.model.kl_per_layer):
            self.log(f"kl_loss_z{i+1}", kl_loss.item(), **log_vars)
        self.log(f"kl_loss_total", weighted_kl.sum().item(), **log_vars)
        self.log("train_loss", total_loss.item(), **log_vars)
        self.log("kl weight (z1)", self.current_kl_weights[0], **log_vars)
        self.log("kl weight (z2)", self.current_kl_weights[1], **log_vars)

        lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", lr, **log_vars)
        return total_loss

    def configure_optimizers(self):
        '''Define optimizer'''
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_init)

        lr_lambda = lambda epoch: set_lr(self.config, epoch) / self.config.lr_init
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def on_train_epoch_start(self):
        '''Update the KL weight based on the current epoch'''
        self.current_kl_weights = set_kl_weight(self.config, self.current_epoch)

    def on_load_checkpoint(self, checkpoint):
        """Fix the checkpoint loading issue for deepspeed."""
        if "state_dict" in checkpoint:
            return
        state_dict = checkpoint['module']
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        checkpoint['state_dict'] = state_dict
        return
    

# self.lr = set_lr(self.config, self.current_epoch)