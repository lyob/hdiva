import torch
from lightning.pytorch import LightningModule
from dataclasses import asdict
from utils.model_init import init_ddpm_model
from utils.training import set_lr


# Define the Lightning Module
class DDPM_Lightning(LightningModule):
    def __init__(self, config):
        super().__init__()

        # ------------------------------ training params ----------------------------- #
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.model = init_ddpm_model(config)

        self.strict_loading = False

    def training_step(self, batch, batch_idx):
        mse_loss = self.model.compute_loss(batch)

        log_vars = dict(
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        self.log("mse_loss", mse_loss.item(), **log_vars)

        lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", lr, **log_vars)
        return mse_loss

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_init)
        lr_lambda = lambda epoch: set_lr(self.config, epoch) / self.config.lr_init
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # Update the learning rate at every step
            "frequency": 1,  # Update every step
        }
        return [optimizer], [scheduler_config]

    def on_load_checkpoint(self, checkpoint):
        """Fix the checkpoint loading issue for deepspeed."""
        if "state_dict" in checkpoint:
            return
        state_dict = checkpoint['module']
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        checkpoint['state_dict'] = state_dict
        return