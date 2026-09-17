import os
import json
import re
import sys

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from dataclasses import asdict

notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from a_datasets.ring.data import RingDataset
from a_datasets.ellipse.data import EllipseDataset
from torch.utils.data import DataLoader
from b_models.configs.sami_simple_config import Config
from b_models.base_modules.mlp import MLPEncoder, MLPDenoiser
from b_models.sami.sami_module import SAMI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################
config = Config()

save_loc = f"{parent_dir}/c_training/local_weights/{config.dataset_name}"
os.makedirs(save_loc, exist_ok=True)

if config.dataset_name == "ring":
    dataset = RingDataset(n_samples=config.num_samples, noise=config.noise)
elif config.dataset_name == "ellipse":
    dataset = EllipseDataset(n_samples=config.num_samples, noise=config.noise)
print(f'dataset name is {config.dataset_name}.')
data = dataset.data
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


# a list of betas trains (and saves) one model per beta, a float trains a single model
betas = (
    [float(b) for b in config.beta_init]
    if isinstance(config.beta_init, (list, tuple))
    else [float(config.beta_init)]
)

# optionally initialize the denoiser from a separately pretrained denoiser run
denoiser_state = None
if config.pretrained_denoiser is not None:
    denoiser_num = f"{int(config.pretrained_denoiser):03d}"
    denoiser_ckpt = f"{save_loc}/sami_denoiser_weights_v{denoiser_num}.pth"
    print(f"loading pretrained denoiser version {denoiser_num}")

    # warn if the pretrained run used a different denoiser architecture
    denoiser_config_path = f"{save_loc}/sami_denoiser_config_v{denoiser_num}.json"
    if os.path.exists(denoiser_config_path):
        with open(denoiser_config_path) as f:
            pretrained_config = json.load(f)
        denoiser_fields = [
            "input_dim", "denoiser_type", "parameterization", "denoiser_act_fn",
            "bias", "time_embedding_method", "hidden_dim", "num_layers", "time_channels",
        ]
        mismatched = {
            k: (pretrained_config[k], getattr(config, k))
            for k in denoiser_fields
            if k in pretrained_config and pretrained_config[k] != getattr(config, k)
        }
        if mismatched:
            print(f"WARNING: denoiser config mismatch (pretrained, current): {mismatched}")

    # the checkpoint is a DDPM state_dict, so the denoiser weights sit under "denoiser."
    ckpt = torch.load(denoiser_ckpt, map_location="cpu")
    denoiser_state = {
        k[len("denoiser."):]: v for k, v in ckpt.items() if k.startswith("denoiser.")
    }

for beta in betas:
    if len(betas) > 1:
        print(f"\n=== training with beta = {beta} ===")

    # each beta gets a freshly initialized model
    denoiser = MLPDenoiser(config)
    infnet = MLPEncoder(config)
    if denoiser_state is not None:
        denoiser.load_state_dict(denoiser_state)

    model = SAMI(
        denoiser,
        infnet,
        config.noise_schedule,
        config.parameterization,
        config.sigma_minmax,
        config.timestep_dist,
        config.num_timesteps,
        config.reduction,
        config.rate_type,
        config.weighted_mse,
        config.weighted_rate,
    )

    # now train model
    model.train()
    model.to(device)

    # optionally hold the denoiser fixed and only train the inference network
    if config.freeze_denoiser:
        print("freezing denoiser weights")
        for p in model.denoiser.parameters():
            p.requires_grad_(False)
        model.denoiser.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.lr)
    n_epochs = config.num_epochs

    for epoch in tqdm(range(int(n_epochs))):
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_rate_loss = 0.0
        for batch in dataloader:
            # print(batch['data'])
            batch = batch['data'].to(device)
            optimizer.zero_grad()
            loss, mse_loss, rate_loss = model.compute_loss(batch, beta=beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_rate_loss += rate_loss.item()
        epoch_loss /= len(dataloader)
        epoch_mse_loss /= len(dataloader)
        epoch_rate_loss /= len(dataloader)
        if epoch % config.log_every_n_epochs == 0:
            print(
                f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, "
                f"MSE Loss: {epoch_mse_loss:.4f}, Rate Loss: {epoch_rate_loss:.4f}"
            )

    # save weights and config
    # figure out model num: one past the largest existing version, else 1
    existing_nums = [
        int(m.group(1))
        for f in os.listdir(save_loc)
        if (m := re.fullmatch(r"sami_weights_v(\d+)\.pth", f))
    ]
    model_num = f"{max(existing_nums, default=0) + 1:03d}"
    print(f"saving model version {model_num}")

    # save weights
    torch.save(model.state_dict(), f"{save_loc}/sami_weights_v{model_num}.pth")

    # save the config too as a dict, with the beta this run actually used
    config_dict = asdict(config)
    config_dict["beta_init"] = beta
    with open(f"{save_loc}/sami_config_v{model_num}.json", "w") as f:
        json.dump(config_dict, f, indent=4)