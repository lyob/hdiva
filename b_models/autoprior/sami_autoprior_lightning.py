from dataclasses import asdict

import lightning as L

from b_models.autoprior.sami_autoprior import (get_autoprior_weights_path,
                                               init_sami_autoprior_model,
                                               load_autoprior_weights_into_sami)
from b_models.sami.sami_lightning import SAMI_Lightning


class SAMI_AutoPrior_Lightning(SAMI_Lightning):
    '''SAMI_Lightning with the autoprior denoiser/encoder, initialized from pretrained autoprior weights.

    Only __init__ differs; the training step, beta schedule and optimizers come from SAMI_Lightning.
    '''
    def __init__(self, config):
        L.LightningModule.__init__(self)
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.model = init_sami_autoprior_model(config)

        # when resuming from a lightning checkpoint, load_from_checkpoint overwrites these weights anyway
        if config.load_autoprior_weights:
            weights_path = get_autoprior_weights_path(config)
            self.model = load_autoprior_weights_into_sami(self.model, weights_path)

        if config.train_infnet_only:
            for param in self.model.denoiser.parameters():
                param.requires_grad = False

        self.strict_loading = False
