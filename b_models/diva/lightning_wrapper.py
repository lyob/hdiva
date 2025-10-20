import torch
import lightning as L
from dataclasses import asdict
from utils.config_utils import init_diva_model, load_pretrained_module_from_wandb
from utils.training import set_lr, set_kl_weight

# Define the Lightning Module
class DiVA_Lightning(L.LightningModule):
    def __init__(self, config, pretrained_wandb_config=None):
        super().__init__()

        # ------------------------------ training params ----------------------------- #
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.model = init_diva_model(config)
        
        # -------------------------- freeze / train denoiser ------------------------- #
        if config.use_pretrained_denoiser_only:
            # Load pretrained denoiser weights from wandb if not resuming from checkpoint
            if pretrained_wandb_config:
                self.model = load_pretrained_module_from_wandb(pretrained_wandb_config, new_model=self.model, module_name='denoiser')
            
            if config.train_infnet_only:
                # Freeze the weights of the denoiser
                self.model.denoiser.requires_grad = False

                for param in self.model.denoiser.parameters():
                    param.requires_grad = False

        self.strict_loading = False

    def training_step(self, batch, batch_idx):
        total_loss, mse_loss, kl_loss = self.model.compute_loss(batch, self.current_kl_weight)  # Define compute_loss in diva

        log_vars = dict(
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        self.log("train_loss", total_loss.item(), **log_vars)
        self.log("mse_loss", mse_loss.item(), **log_vars)
        self.log("kl_loss", kl_loss.item(), **log_vars)
        self.log("current_kl_weight", self.current_kl_weight, **log_vars)

        lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", lr, **log_vars)

        return total_loss

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

    def on_train_epoch_start(self):
        # Update the KL weight based on the current epoch
        self.current_kl_weight = set_kl_weight(self.config, self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        # if self.config.train_infnet_only:
        #     # remove the denoiser from the checkpoint
        #     checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if "denoiser" not in k}
        #     return
        # else:
        #     return
        # # remove the denoiser weights from the checkpoint
        # checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if "denoiser" not in k}
        return

    def on_load_checkpoint(self, checkpoint):
        """Fix the checkpoint loading issue for deepspeed."""
        if "state_dict" in checkpoint:
            return
        state_dict = checkpoint['module']
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        checkpoint['state_dict'] = state_dict
        return