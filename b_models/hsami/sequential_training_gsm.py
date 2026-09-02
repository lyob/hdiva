import dataclasses
import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from a_datasets.gsm.data import GSMDataset
from a_datasets.ring.data_flat import WrappedPhaseRingFlat
from b_models.configs.hsami_config import HSAMI_Training_Config
from b_models.hsami.denoiser import Denoiser
from b_models.hsami.hsami_alternate import HSAMI
from b_models.hsami.recnet import RecNet

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class SequentialTrainingConfig(HSAMI_Training_Config):
    # Which stages to run (e.g. [1, 2] or [2] to resume from a prior run)
    stages: list = field(default_factory=lambda: [1, 2])
    # Run name to resume from when the first stage > 1 (e.g. "sequential_20260618_015523")
    resume_from_run: str = "sequential_20260618_160632"

    # Stage 1: unconditional DDPM on x
    stage1_epochs: int = 500
    stage1_den_x_lr_init: float = 2e-4
    stage1_den_x_lr_final: float = 2e-5
    stage1_den_x_warmup_epochs: int = 50

    # Stage 2: denoiser_x + x→y encoder, guided x MSE loss
    stage2_epochs: int = 2000
    stage2_den_x_lr: float = 1e-5
    stage2_enc_xy_lr: float = 1e-5
    stage2_beta_y_rate: float = 0.0  # penalty on ||score_y_given_xg||; 0 = off

    # Stage 3: unconditional denoiser_y on frozen y representation
    # (encoder + denoiser_x stay requires_grad=True but get no gradients from this loss)
    stage3_epochs: int = 1000
    stage3_den_y_lr_init: float = 2e-4
    stage3_den_y_lr_final: float = 2e-5
    stage3_den_y_warmup_epochs: int = 50

    # Stage 4: full training — adds y→z encoder, trains all components jointly
    stage4_epochs: int = 6000
    stage4_den_x_lr: float = 1e-5
    stage4_enc_xy_lr: float = 1e-5
    stage4_den_y_lr: float = 1e-5
    stage4_enc_yz_lr: float = 1e-4
    stage4_beta_y: float = 1e-5
    stage4_beta_z: float = 1e-5
    stage4_t_y_curriculum_steps: int = 100000
    stage4_t_y_min_sqrt_alpha_bar: float = 0.1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class Tee:
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


def save_sequential_config(config, run_dir):
    parent_field_names = {f.name for f in dataclasses.fields(HSAMI_Training_Config)}
    own_keys = {f.name for f in dataclasses.fields(SequentialTrainingConfig)} - parent_field_names
    d = {k: v for k, v in dataclasses.asdict(config).items() if k in own_keys}
    with open(os.path.join(run_dir, "sequential_config.json"), "w") as f:
        json.dump(d, f, indent=2)


# Keys (by state-dict prefix) that are trained *through* each stage.
# Loading from a stage-N checkpoint restores only these; everything else
# stays at fresh random init so the next stage starts clean.
_TRAINED_PREFIXES = {
    1: (
        "denoiser_x.",
    ),
    2: (
        "denoiser_x.",
        "recnet.layers_x_to_y.", "recnet.head_y.",
    ),
    3: (
        "denoiser_x.",
        "recnet.layers_x_to_y.", "recnet.head_y.",
        "denoiser_y.",
    ),
    4: (
        "denoiser_x.",
        "recnet.layers_x_to_y.", "recnet.head_y.",
        "denoiser_y.",
        "recnet.input_head_y.", "recnet.layers_y_to_z.", "recnet.head_z.",
    ),
}


def load_stage_checkpoint(hsami, checkpoint_dir, run_name, stage, device):
    run_dir = os.path.join(checkpoint_dir, run_name)
    checkpoints = sorted(glob.glob(os.path.join(run_dir, f"stage{stage}_epoch_*.pt")))
    if not checkpoints:
        raise FileNotFoundError(f"No stage {stage} checkpoints found in {run_dir}")
    path = checkpoints[-1]
    saved_sd = torch.load(path, map_location=device)["model_state_dict"]

    prefixes = _TRAINED_PREFIXES[stage]
    current_sd = hsami.state_dict()
    loaded_keys = [k for k in saved_sd if any(k.startswith(p) for p in prefixes)]
    for k in loaded_keys:
        current_sd[k] = saved_sd[k]
    hsami.load_state_dict(current_sd)

    print(f"loaded checkpoint: {path}")
    print(f"  restored {len(loaded_keys)} tensors "
          f"(prefixes: {', '.join(prefixes)})")


def get_lr(epoch, warmup_epochs, num_epochs, lr_init, lr_final):
    """Linear warmup + cosine decay."""
    if epoch < warmup_epochs:
        return lr_init * epoch / max(warmup_epochs, 1)
    t = epoch - warmup_epochs
    T = max(num_epochs - warmup_epochs, 1)
    frac = 0.5 * (1 + math.cos(math.pi * t / T))
    return lr_final + frac * (lr_init - lr_final)


def get_t_y_max(step, t_y_hard_max, num_timesteps_y, curriculum_steps):
    """Linearly open the t_y range from t_y_hard_max to num_timesteps_y-1."""
    if curriculum_steps <= 0 or step >= curriculum_steps:
        return num_timesteps_y - 1
    progress = step / curriculum_steps
    return int(round(t_y_hard_max + progress * (num_timesteps_y - 1 - t_y_hard_max)))


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
    return HSAMI(recnet=recnet, denoiser_x=denoiser_x, denoiser_y=denoiser_y, args=config).to(device)


def _enc_xy_params(hsami):
    return (
        list(hsami.recnet.layers_x_to_y.parameters())
        + list(hsami.recnet.head_y.parameters())
    )


def _enc_yz_params(hsami):
    return (
        list(hsami.recnet.input_head_y.parameters())
        + list(hsami.recnet.layers_y_to_z.parameters())
        + list(hsami.recnet.head_z.parameters())
    )


def save_checkpoint(hsami, run_dir, stage, epoch, optimizer=None, step=None):
    payload = {"stage": stage, "epoch": epoch, "model_state_dict": hsami.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if step is not None:
        payload["step"] = step
    path = os.path.join(run_dir, f"stage{stage}_epoch_{epoch:06d}.pt")
    torch.save(payload, path)
    print(f"saved checkpoint: {path}")


# ---------------------------------------------------------------------------
# stage 2 loss
# ---------------------------------------------------------------------------

def compute_loss_stage2(hsami, x, beta_y_rate=0.0):
    """Guided x MSE: ∇_{xg} log p(y_clean | xg) guides the x denoiser.

    y_clean is sampled from the encoder on clean x and treated as a fixed
    observation. Gradients flow to denoiser_x and recnet.{layers_x_to_y, head_y}
    via the score computation (create_graph=True inside compute_score_from_logp).

    Returns:
        total_loss, mse_loss, score_xg_norm, score_y_xg_norm
        The last two are detached scalars for logging (L2 norm per sample, batch-averaged).
    """
    B = x.shape[0]
    device = x.device
    clean_x = x.detach()

    # sample y from clean x
    mu_y_from_x, logvar_y_from_x = hsami.recnet.x_to_y(clean_x)
    y_sample_up = hsami.recnet.sample(mu_y_from_x, logvar_y_from_x)

    # noisy x
    t_tensor_x = hsami.denoiser_x.sample_timesteps(
        B, hsami.denoiser_x.num_timesteps,
        dist=hsami.denoiser_x.timestep_dist, device=device,
    )
    noisy_x, noise_x = hsami.denoiser_x.make_noisy(clean_x, t_tensor_x)
    noisy_x = noisy_x.requires_grad_(True)
    gamma = hsami.extract(hsami.denoiser_x.one_minus_alpha_bars, t_tensor_x, clean_x.shape)

    # encoder + denoiser on noisy x
    mu_y_from_xg, logvar_y_from_xg = hsami.recnet.x_to_y(noisy_x)
    pred_noise_x = hsami.denoiser_x.denoiser(noisy_x, t_tensor_x)

    # guidance score ∇_{xg} log p(y_clean | xg), then guided noise prediction
    score_y_given_xg = hsami.compute_score_y_given_xg(
        mu_y_from_xg, logvar_y_from_xg, y_sample_up, noisy_x,
    )
    pred_noise_x_guided = pred_noise_x - gamma.sqrt() * score_y_given_xg

    mse_loss = F.mse_loss(pred_noise_x_guided, noise_x)
    rate_y = score_y_given_xg.norm(dim=-1).mean()
    total_loss = mse_loss + beta_y_rate * rate_y

    score_xg_norm = pred_noise_x.detach().norm(dim=-1).mean()
    score_y_xg_norm = score_y_given_xg.detach().norm(dim=-1).mean()
    return total_loss, mse_loss, score_xg_norm, score_y_xg_norm


# ---------------------------------------------------------------------------
# stage training functions
# ---------------------------------------------------------------------------

def train_stage1(hsami, loader, config, device, run_dir):
    """Stage 1: unconditional DDPM on x, trains denoiser_x only."""
    optimizer = torch.optim.Adam(
        hsami.denoiser_x.parameters(), lr=config.stage1_den_x_lr_init,
    )
    print(f"=== Stage 1: denoiser_x unconditional ({config.stage1_epochs} epochs) ===")
    log_every = 20
    step = 0
    print_flag = False

    for epoch in range(config.stage1_epochs):
        if epoch % log_every == 0:
            print_flag = True
        lr = get_lr(epoch, config.stage1_den_x_warmup_epochs, config.stage1_epochs,
                    config.stage1_den_x_lr_init, config.stage1_den_x_lr_final)
        optimizer.param_groups[0]["lr"] = lr

        for (_, _), x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = hsami.denoiser_x.compute_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hsami.denoiser_x.parameters(), 1.0)
            optimizer.step()
            if print_flag:
                print(
                    f"[stage1] epoch {epoch:5d} | step {step:7d} | "
                    f"loss {loss.item():.4f} | lr {lr:.2e}"
                )
                print_flag = False
            step += 1

        if (epoch + 1) % config.checkpoint_every_n_epochs == 0:
            save_checkpoint(hsami, run_dir, stage=1, epoch=epoch + 1)

    save_checkpoint(hsami, run_dir, stage=1, epoch=config.stage1_epochs)
    print("=== Stage 1 complete ===")


def train_stage2(hsami, loader, config, device, run_dir):
    """Stage 2: denoiser_x + x→y encoder trained jointly via guided x MSE.

    The encoder learns y representations that improve x denoiser guidance.
    denoiser_x continues training; stage 1 weights are NOT frozen.
    """
    enc_xy = _enc_xy_params(hsami)
    optimizer = torch.optim.Adam([
        {"params": hsami.denoiser_x.parameters(), "lr": config.stage2_den_x_lr},
        {"params": enc_xy, "lr": config.stage2_enc_xy_lr},
    ])
    print(f"=== Stage 2: denoiser_x + x→y encoder ({config.stage2_epochs} epochs) ===")
    log_every = 20
    step = 0
    print_flag = False

    for epoch in range(config.stage2_epochs):
        if epoch % log_every == 0:
            print_flag = True

        for (_, _), x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss, mse_loss, score_xg_norm, score_y_xg_norm = compute_loss_stage2(
                hsami, x, beta_y_rate=config.stage2_beta_y_rate,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(hsami.denoiser_x.parameters()) + enc_xy, 1.0,
            )
            optimizer.step()
            if print_flag:
                print(
                    f"[stage2] epoch {epoch:5d} | step {step:7d} | "
                    f"mse_x_guided {mse_loss.item():.4f} | "
                    f"|s(xg)| {score_xg_norm.item():.4e} | "
                    f"|s(y|xg)| {score_y_xg_norm.item():.4e}"
                )
                print_flag = False
            step += 1

        if (epoch + 1) % config.checkpoint_every_n_epochs == 0:
            save_checkpoint(hsami, run_dir, stage=2, epoch=epoch + 1)

    save_checkpoint(hsami, run_dir, stage=2, epoch=config.stage2_epochs)
    print("=== Stage 2 complete ===")


def train_stage3(hsami, loader, config, device, run_dir):
    """Stage 3: unconditional denoiser_y on y representations from the encoder.

    y samples are drawn from the (fixed) encoder under no_grad — the denoiser_y
    loss (denoiser_y.compute_loss) detaches its input internally, so no gradients
    reach the encoder or denoiser_x. Those components remain requires_grad=True
    (unfrozen) but are not updated in this stage.
    """
    optimizer = torch.optim.Adam(
        hsami.denoiser_y.parameters(), lr=config.stage3_den_y_lr_init,
    )
    print(f"=== Stage 3: denoiser_y unconditional ({config.stage3_epochs} epochs) ===")
    log_every = 20
    step = 0
    print_flag = False

    for epoch in range(config.stage3_epochs):
        if epoch % log_every == 0:
            print_flag = True
        lr = get_lr(epoch, config.stage3_den_y_warmup_epochs, config.stage3_epochs,
                    config.stage3_den_y_lr_init, config.stage3_den_y_lr_final)
        optimizer.param_groups[0]["lr"] = lr

        for (_, _), x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                mu_y, logvar_y = hsami.recnet.x_to_y(x)
                y_sample = hsami.recnet.sample(mu_y, logvar_y)
            loss = hsami.denoiser_y.compute_loss(y_sample)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hsami.denoiser_y.parameters(), 1.0)
            optimizer.step()
            if print_flag:
                print(
                    f"[stage3] epoch {epoch:5d} | step {step:7d} | "
                    f"loss_y {loss.item():.4f} | lr {lr:.2e}"
                )
                print_flag = False
            step += 1

        if (epoch + 1) % config.checkpoint_every_n_epochs == 0:
            save_checkpoint(hsami, run_dir, stage=3, epoch=epoch + 1)

    save_checkpoint(hsami, run_dir, stage=3, epoch=config.stage3_epochs)
    print("=== Stage 3 complete ===")


def train_stage4(hsami, loader, config, device, run_dir):
    """Stage 4: full joint training.

    Adds y→z encoder (input_head_y, layers_y_to_z, head_z) to the optimizer.
    All previously trained components remain unfrozen and continue updating.
    Uses the complete HSAMI loss: mse_x + mse_y + beta_y * rate_y + beta_z * rate_z.
    """
    enc_xy = _enc_xy_params(hsami)
    enc_yz = _enc_yz_params(hsami)
    optimizer = torch.optim.Adam([
        {"params": hsami.denoiser_x.parameters(), "lr": config.stage4_den_x_lr},
        {"params": enc_xy, "lr": config.stage4_enc_xy_lr},
        {"params": hsami.denoiser_y.parameters(), "lr": config.stage4_den_y_lr},
        {"params": enc_yz, "lr": config.stage4_enc_yz_lr},
    ])

    # t_y curriculum: cap maximum noise level at start, open up gradually
    if config.stage4_t_y_min_sqrt_alpha_bar > 0 and config.stage4_t_y_curriculum_steps > 0:
        mask = hsami.denoiser_y.sqrt_alpha_bars >= config.stage4_t_y_min_sqrt_alpha_bar
        t_y_hard_max = int(mask.nonzero(as_tuple=False)[-1].item()) if mask.any() else 0
        print(
            f"t_y curriculum: hard_max={t_y_hard_max} "
            f"(sqrt_ab={config.stage4_t_y_min_sqrt_alpha_bar}), "
            f"full range in {config.stage4_t_y_curriculum_steps} steps"
        )
    else:
        t_y_hard_max = config.num_timesteps_y - 1

    print(f"=== Stage 4: full training ({config.stage4_epochs} epochs) ===")
    log_every = 20
    step = 0
    print_flag = False

    for epoch in range(config.stage4_epochs):
        if epoch % log_every == 0:
            print_flag = True

        for (_, _), x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            t_y_max = get_t_y_max(
                step, t_y_hard_max, config.num_timesteps_y,
                config.stage4_t_y_curriculum_steps,
            )
            total_loss, mse_x, mse_y, rate_y_loss, rate_z_loss, debug_norms = hsami.compute_loss(
                x,
                beta_y=config.stage4_beta_y,
                beta_z=config.stage4_beta_z,
                t_y_max=t_y_max,
                debug=print_flag,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(hsami.parameters(), 1.0)
            optimizer.step()
            if print_flag:
                print(
                    f"[stage4] epoch {epoch:5d} | step {step:7d} | "
                    f"loss {total_loss.item():.4f} | mse_x {mse_x.item():.4f} | "
                    f"mse_y {mse_y.item():.4f} | rate_y {rate_y_loss.item():.4f} | "
                    f"rate_z {rate_z_loss.item():.4f} | "
                    f"t_y_max {t_y_max}/{config.num_timesteps_y - 1}"
                )
                if debug_norms:
                    print("  norms: " + " | ".join(f"{k} {v:.3e}" for k, v in debug_norms.items()))
                print_flag = False
            step += 1

        if (epoch + 1) % config.checkpoint_every_n_epochs == 0:
            save_checkpoint(
                hsami, run_dir, stage=4, epoch=epoch + 1,
                optimizer=optimizer, step=step,
            )

    save_checkpoint(
        hsami, run_dir, stage=4, epoch=config.stage4_epochs,
        optimizer=optimizer, step=step,
    )
    print("=== Stage 4 complete ===")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def train():
    config = SequentialTrainingConfig()
    torch.manual_seed(config.training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

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

    hsami = build_model(config, device)
    num_params = sum(p.numel() for p in hsami.parameters() if p.requires_grad)
    print(f"parameters: {num_params:,}")

    checkpoint_dir = os.path.join(project_root, "c_training", "hsami_checkpoints")
    run_dir = os.path.join(
        checkpoint_dir, "sequential_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)
    save_config(config, run_dir)
    save_sequential_config(config, run_dir)

    tee = Tee(os.path.join(run_dir, "train.out"))
    sys.stdout = tee
    print(f"run dir: {run_dir}")

    stages = sorted(config.stages)
    first_stage = stages[0]
    if first_stage > 1:
        if not config.resume_from_run:
            raise ValueError(
                f"stages starts at {first_stage} but resume_from_run is not set"
            )
        load_stage_checkpoint(
            hsami, checkpoint_dir, config.resume_from_run, first_stage - 1, device
        )

    stage_fns = {
        1: train_stage1,
        2: train_stage2,
        3: train_stage3,
        4: train_stage4,
    }
    for stage in stages:
        stage_fns[stage](hsami, loader, config, device, run_dir)

    tee.close()


if __name__ == "__main__":
    train()
