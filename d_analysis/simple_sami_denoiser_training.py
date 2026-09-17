import os
import sys
import re
import json
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
from b_models.ddpm.ddpm_module import DDPM

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

# dataset = RingDataset(n_samples=config.num_samples, noise=config.noise)
# data = dataset.data
# dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


denoiser = MLPDenoiser(config)
infnet = MLPEncoder(config)
model = DDPM(
    denoiser,
    config.noise_schedule,
    config.parameterization,
    config.sigma_minmax,
    config.timestep_dist,
    config.num_timesteps,
    config.reduction,
    config.weighted_mse,
)

# now train model
model.train()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
n_epochs = config.num_epochs
beta = config.beta_init

for epoch in tqdm(range(int(n_epochs))):
    epoch_loss = 0.0
    for batch in dataloader:
        batch = batch['data'].to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(dataloader)
    if epoch % config.log_every_n_epochs == 0:
        print(
            f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}"
        )

# save the weights
save_loc = f"{parent_dir}/c_training/local_weights/{config.dataset_name}"
os.makedirs(save_loc, exist_ok=True)


# figure out model num: one past the largest existing version, else 1
existing_nums = [
    int(m.group(1))
    for f in os.listdir(save_loc)
    if (m := re.fullmatch(r"sami_denoiser_weights_v(\d+)\.pth", f))
]
model_num = f"{max(existing_nums, default=0) + 1:03d}"
print(f"saving model version {model_num}")

torch.save(model.state_dict(), f"{save_loc}/sami_denoiser_weights_v{model_num}.pth")

# save the config too as a dict
config_dict = asdict(config)
with open(f"{save_loc}/sami_denoiser_config_v{model_num}.json", "w") as f:
    json.dump(config_dict, f, indent=4)