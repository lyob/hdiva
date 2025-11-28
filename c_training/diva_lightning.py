import torch
import lightning as L
from dataclasses import asdict
from utils.wandb_utils import load_pretrained_module_from_wandb
from utils.model_init import init_diva_model
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

        lrs = [pg['lr'] for pg in self.optimizers().param_groups]
        self.log("learning_rate", lrs[0], **log_vars)
        if len(lrs) > 1:
            self.log("lr_encoder", lrs[1], **log_vars)

        return total_loss

    def configure_optimizers(self):
        # Define optimizer with parameter groups
        # Filter parameters that require gradients
        denoiser_params = [p for p in self.model.denoiser.parameters() if p.requires_grad]
        infnet_params = [p for p in self.model.infnet.parameters() if p.requires_grad]

        optimizer_groups = []
        lr_lambdas = []

        # Denoiser group
        if denoiser_params:
            optimizer_groups.append({'params': denoiser_params, 'lr': self.config.lr_init})
            lr_lambdas.append(lambda epoch: set_lr(self.config, epoch) / self.config.lr_init)

        # Encoder (InfNet) group
        if infnet_params:
            optimizer_groups.append({'params': infnet_params, 'lr': self.config.encoder_lr_init})
            
            def encoder_schedule(epoch):
                if epoch < self.config.encoder_warmup_epochs:
                    return 1.0
                elif epoch < self.config.encoder_warmup_epochs + self.config.encoder_convergence_epochs:
                    # Convergence phase
                    progress = (epoch - self.config.encoder_warmup_epochs) / self.config.encoder_convergence_epochs
                    target_lr = self.config.encoder_lr_init + progress * (self.config.encoder_lr_final - self.config.encoder_lr_init)
                    return target_lr / self.config.encoder_lr_init
                else:
                    # Converged to encoder_lr_final
                    return self.config.encoder_lr_final / self.config.encoder_lr_init
            
            lr_lambdas.append(encoder_schedule)

        optimizer = torch.optim.Adam(optimizer_groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
        
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