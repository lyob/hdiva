import dataclasses
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

# resolve project root (two levels up from b_models/hsami/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from a_datasets.gsm.data import GSMDataset
from b_models.configs.hsami_config import HSAMI_Training_Config
from b_models.hsami.denoiser import Denoiser
from b_models.hsami.hsami_alternate import HSAMI
from b_models.hsami.recnet import RecNet

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class Tee:
    """Write to both a file and the original stdout simultaneously."""
    def __init__(self, path):
        self._file = open(path, "w", buffering=1)
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


def save_config(config, run_dir):
    d = dataclasses.asdict(config)
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(d, f, indent=2)


def get_t_y_max(step, t_y_hard_max, num_timesteps_y, curriculum_steps):
    """Linearly open the t_y range from t_y_hard_max to num_timesteps_y-1."""
    if curriculum_steps <= 0 or step >= curriculum_steps:
        return num_timesteps_y - 1
    progress = step / curriculum_steps
    return int(round(t_y_hard_max + progress * (num_timesteps_y - 1 - t_y_hard_max)))


def get_beta(epoch, config):
    if config.beta_schedule == "wait_then_anneal":
        """Linear/cosine beta annealing with a flat wait period."""
        if epoch < config.beta_wait_epochs:
            return config.beta_init
        t = min(epoch - config.beta_wait_epochs, config.beta_annealing_epochs)
        frac = t / config.beta_annealing_epochs
        if config.beta_annealing_schedule == "cosine":
            frac = 0.5 * (1 - math.cos(math.pi * frac))
        return config.beta_init + frac * (config.beta_final - config.beta_init)
    elif config.beta_schedule == "constant":
        return config.beta_init



def get_lr(epoch, lr_num_warmup_epochs, num_epochs, lr_init, lr_final):
    """Linear warmup + cosine decay."""
    if epoch < lr_num_warmup_epochs:
        return lr_init * epoch / max(lr_num_warmup_epochs, 1)
    t = epoch - lr_num_warmup_epochs
    T = max(num_epochs - lr_num_warmup_epochs, 1)
    frac = 0.5 * (1 + math.cos(math.pi * t / T))
    return lr_final + frac * (lr_init - lr_final)


def build_model(config, device):
    recnet = RecNet(config).to(device)
    denoiser_x = Denoiser(
        config.input_dim,
        hidden_dim=config.hidden_dim_x,
        num_blocks=config.num_blocks_x,
        time_dim=config.time_dim_x,
        activation=config.activation,
    ).to(device)
    denoiser_y = Denoiser(
        config.latent_dim_y,
        hidden_dim=config.hidden_dim_y,
        num_blocks=config.num_blocks_y,
        time_dim=config.time_dim_y,
        activation=config.activation,
    ).to(device)
    hsami = HSAMI(recnet=recnet, denoiser_x=denoiser_x, denoiser_y=denoiser_y, args=config).to(device)
    return hsami


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def pretrain_denoiser_x(hsami, loader, config, device, checkpoint_dir):
    """Phase 1: train denoiser_x unconditionally with standard DDPM loss."""
    optimizer = torch.optim.Adam(
        hsami.denoiser_x.parameters(), lr=config.den_x_lr_init
    )

    step = 0
    log_every = 20
    print_flag = False
    num_epochs = config.pretrain_denoiser_x_epochs

    print(f"=== denoiser_x pretraining for {num_epochs} epochs ===")
    for epoch in range(num_epochs):
        if epoch % log_every == 0:
            print_flag = True

        lr = get_lr(epoch, config.den_x_lr_num_warmup_epochs, num_epochs, config.den_x_lr_init, config.den_x_lr_final)
        optimizer.param_groups[0]["lr"] = lr

        for (_, _), x in loader:
            x = x.to(device)
            optimizer.zero_grad()

            loss = hsami.denoiser_x.compute_loss(x)
            loss.backward()
            optimizer.step()

            if print_flag:
                print(
                    f"[pretrain] epoch {epoch:5d} | step {step:7d} | "
                    f"loss {loss.item():.4f} | lr {lr:.2e}"
                )
                print_flag = False
            step += 1

    ckpt_path = os.path.join(checkpoint_dir, "denoiser_x_pretrain.pt")
    torch.save({"epoch": num_epochs, "denoiser_x_state_dict": hsami.denoiser_x.state_dict()}, ckpt_path)
    print(f"saved denoiser_x pretrain checkpoint: {ckpt_path}")
    print("=== denoiser_x pretraining complete, continuing with full training ===")


def train():
    config = HSAMI_Training_Config()
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # dataset
    dataset = GSMDataset(
        num_samples=config.dataset_size,
        sigma_z=config.sigma_z,
        lambdas=config.lambdas,
        A=config.A,
        sigma_x=config.sigma_x,
        seed=config.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train_batch_size_per_gpu,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    # model
    hsami = build_model(config, device)
    num_params = sum(p.numel() for p in hsami.parameters() if p.requires_grad)
    print(f"parameters: {num_params:,}")

    checkpoint_dir = os.path.join(project_root, "c_training", "hsami_checkpoints")
    run_dir = os.path.join(checkpoint_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    save_config(config, run_dir)
    tee = Tee(os.path.join(run_dir, "train.out"))
    sys.stdout = tee
    print(f"run dir: {run_dir}")

    # precompute t_y_hard_max: last index where sqrt(alpha_bar_y) >= t_y_min_sqrt_alpha_bar
    if config.t_y_min_sqrt_alpha_bar > 0 and config.t_y_curriculum_steps > 0:
        mask = hsami.denoiser_y.sqrt_alpha_bars >= config.t_y_min_sqrt_alpha_bar
        t_y_hard_max = int(mask.nonzero(as_tuple=False)[-1].item()) if mask.any() else 0
        print(f"t_y curriculum: hard_max={t_y_hard_max} (sqrt_ab={config.t_y_min_sqrt_alpha_bar}), "
              f"full range in {config.t_y_curriculum_steps} steps")
    else:
        t_y_hard_max = config.num_timesteps_y - 1

    # --- denoiser_x initialisation ---
    if config.use_pretrained_denoiser_x:
        ckpt_path = os.path.join(checkpoint_dir, "denoiser_x_pretrain.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        hsami.denoiser_x.load_state_dict(ckpt["denoiser_x_state_dict"])
        print(f"loaded denoiser_x checkpoint: {ckpt_path}")
    elif config.pretrain_denoiser_x:
        pretrain_denoiser_x(hsami, loader, config, device, run_dir)

    if config.freeze_denoiser_x:
        hsami.denoiser_x.requires_grad_(False)
        print("denoiser_x frozen for phase 2")

    # --- optimizer: exclude denoiser_x when frozen ---
    if config.freeze_denoiser_x:
        optimizer = torch.optim.Adam([
            {"params": hsami.recnet.parameters(), "lr": config.encoder_lr_init},
            {"params": list(hsami.denoiser_y.parameters()), "lr": config.den_y_lr_init},
        ])
        enc_idx, den_y_idx = 0, 1
    else:
        optimizer = torch.optim.Adam([
            {"params": hsami.recnet.parameters(), "lr": config.encoder_lr_init},
            {"params": list(hsami.denoiser_x.parameters()), "lr": config.den_x_lr_init},
            {"params": list(hsami.denoiser_y.parameters()), "lr": config.den_y_lr_init},
        ])
        enc_idx, den_x_idx, den_y_idx = 0, 1, 2

    step = 0
    # log_every = 200  # steps
    log_every = 20  # epochs
    print_flag = False

    for epoch in range(config.num_epochs):
        if epoch % log_every == 0:
            print_flag = True

        beta = get_beta(epoch, config)

        # update lrs
        if not config.freeze_denoiser_x:
            denoiser_x_lr = get_lr(epoch, config.den_x_lr_num_warmup_epochs, config.num_epochs, config.den_x_lr_init, config.den_x_lr_final)
            optimizer.param_groups[den_x_idx]["lr"] = denoiser_x_lr
        denoiser_y_lr = get_lr(epoch, config.den_y_lr_num_warmup_epochs, config.num_epochs, config.den_y_lr_init, config.den_y_lr_final)
        optimizer.param_groups[den_y_idx]["lr"] = denoiser_y_lr

        # update encoder lr (linear warmup from lr_init to intermed, then stays at final)
        enc_wait = config.encoder_wait_epochs
        enc_warmup = config.encoder_warmup_epochs
        if epoch < enc_wait:
            enc_lr = config.encoder_lr_init
        elif epoch < enc_wait + enc_warmup:
            frac = (epoch - enc_wait) / max(enc_warmup, 1)
            enc_lr = config.encoder_lr_init + frac * (config.encoder_lr_intermed - config.encoder_lr_init)
        else:
            enc_lr = config.encoder_lr_final
        optimizer.param_groups[enc_idx]["lr"] = enc_lr

        for (_, _), x in loader:
            x = x.to(device)
            optimizer.zero_grad()

            t_y_max = get_t_y_max(step, t_y_hard_max, config.num_timesteps_y, config.t_y_curriculum_steps)
            total_loss, mse_x, mse_y, rate_y_loss, rate_z_loss, debug_norms = hsami.compute_loss(x, beta_y=beta, beta_z=beta, t_y_max=t_y_max, debug=print_flag)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(hsami.parameters(), max_norm=1.0)
            optimizer.step()

            if print_flag:
                den_x_lr_str = "" if config.freeze_denoiser_x else f" | den_x_lr {denoiser_x_lr:.2e}"
                print(
                    f"epoch {epoch:5d} | step {step:7d} | "
                    f"loss {total_loss.item():.4f} | mse_x {mse_x.item():.4f} | mse_y {mse_y.item():.4f} | "
                    f"rate_y {rate_y_loss.item():.4f} | rate_z {rate_z_loss.item():.4f} | "
                    f"beta {beta:.2e} | t_y_max {t_y_max}/{config.num_timesteps_y - 1} | "
                    f"enc_lr {enc_lr:.2e}{den_x_lr_str} | den_y_lr {denoiser_y_lr:.2e}"
                )
                # --- debug norms (remove when no longer needed) ---
                print("  norms: " + " | ".join(f"{k} {v:.3e}" for k, v in debug_norms.items()))
                # --- end debug norms ---
                print_flag = False
            step += 1

        if (epoch + 1) % config.checkpoint_every_n_epochs == 0:
            ckpt_path = os.path.join(run_dir, f"epoch_{epoch + 1:06d}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": hsami.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
            }, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")

    tee.close()


if __name__ == "__main__":
    train()
