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

from a_datasets.h2disks.data import (
    HierarchicalTwoDiskDistanceIntensityDataset, normalize_to_pm1)
# from a_datasets.hierarchical_disks.data import HierarchicalDiskDataset
from a_datasets.hierarchical_two_disks.data import \
    HierarchicalTwoDiskIntensityDataset
from b_models.configs.hsami_config import HSAMI_Image_Training_Config
from b_models.hsami.denoiser import Denoiser
from b_models.hsami.denoiser_conv import DenoiserConv
from b_models.hsami.hsami_conv import HSAMI
from b_models.hsami.recnet_conv import RecNetConv

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class SequentialTrainingConfig(HSAMI_Image_Training_Config):
    # Which stages to run (e.g. [1, 2] or [2] to resume from a prior run)
    stages: list = field(default_factory=lambda: [4])
    # Run name to resume from when the first stage > 1 (e.g. "sequential_20260618_015523").
    # Used as the default single-checkpoint resume when resume_checkpoints is empty.
    resume_from_run: str = "sequential_20260703_162708"
    # Explicit (stage, run_name) checkpoints to load before resuming, applied in
    # list order — later entries overwrite earlier ones for overlapping components
    # (see _TRAINED_PREFIXES). Lets you combine checkpoints across runs, e.g.
    # [(3, "sequential_A"), (1, "sequential_B")] loads the broad stage-3 state
    # from run A, then overrides just the stage-1 (denoiser_x) weights with run B.
    # If empty, defaults to [(stages[0] - 1, resume_from_run)].
    # resume_checkpoints: list = field(default_factory=list)
    resume_checkpoints: list = field(default_factory=lambda: 
                                     [(1, "sequential_20260703_162708")]
                                     )

    # Stage 1: unconditional DDPM on x
    stage1_epochs: int = 3000
    stage1_den_x_lr_init: float = 3e-4
    stage1_den_x_lr_final: float = 3e-4
    stage1_den_x_warmup_epochs: int = 100

    # Stage 2: denoiser_x + x→y encoder, guided x MSE loss
    stage2_epochs: int = 4000
    stage2_den_x_lr: float = 3e-4
    stage2_enc_xy_lr: float = 3e-4
    stage2_beta_y_rate: float = 1e-3  # penalty on ||score_y_given_xg||; 0 = off

    # Stage 3: unconditional denoiser_y on frozen y representation
    # (encoder + denoiser_x stay requires_grad=True but get no gradients from this loss)
    stage3_epochs: int = 500
    stage3_den_y_lr_init: float = 2e-4
    stage3_den_y_lr_final: float = 2e-5
    stage3_den_y_warmup_epochs: int = 50

    # Stage 4: full training — adds y→z encoder, trains all components jointly
    stage4_epochs: int = 10000
    stage4_den_x_lr: float = 3e-4
    stage4_enc_xy_lr: float = 3e-4
    stage4_den_y_lr: float = 3e-4
    stage4_enc_yz_lr: float = 3e-4
    stage4_beta_y: float = 1e-7          # TARGET (final) beta_y after annealing
    stage4_beta_z: float = 1e-9
    stage4_t_y_curriculum_steps: int = 40000
    stage4_t_y_min_sqrt_alpha_bar: float = 0.1

    # beta_y annealing: hold at init, then ramp to stage4_beta_y. Lets the
    # representation form before the I(X;Y|Z) rate term starts compressing y
    # (without a warmup the rate term over-compresses and q(y|x) blows up).
    stage4_beta_y_init: float = 1e-8            # ~off early
    stage4_beta_y_wait_epochs: int = 200        # hold at init before ramping
    stage4_beta_y_warmup_epochs: int = 1500     # ramp duration (epochs)
    stage4_beta_y_schedule: str = "linear_in_log"  # "linear_in_log" | "cosine" | "linear"


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
    parent_field_names = {f.name for f in dataclasses.fields(HSAMI_Image_Training_Config)}
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
    """Restore model weights trained through `stage` from `run_name`.

    Returns the checkpoint's own (epoch, step, optimizer_state_dict) so that
    callers resuming into the *same* stage can continue training (optimizer
    momentum + step-dependent schedules) instead of restarting it cold.
    Older checkpoints saved without optimizer/step info return None for those.
    """
    run_dir = os.path.join(checkpoint_dir, run_name)
    checkpoints = sorted(glob.glob(os.path.join(run_dir, f"stage{stage}_epoch_*.pt")))
    if not checkpoints:
        raise FileNotFoundError(f"No stage {stage} checkpoints found in {run_dir}")
    path = checkpoints[-1]
    checkpoint = torch.load(path, map_location=device)
    saved_sd = checkpoint["model_state_dict"]

    prefixes = _TRAINED_PREFIXES[stage]
    current_sd = hsami.state_dict()
    loaded_keys = [k for k in saved_sd if any(k.startswith(p) for p in prefixes)]
    for k in loaded_keys:
        current_sd[k] = saved_sd[k]
    hsami.load_state_dict(current_sd)

    print(f"loaded checkpoint: {path}")
    print(f"  restored {len(loaded_keys)} tensors "
          f"(prefixes: {', '.join(prefixes)})")

    return {
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
    }


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


def get_beta_y(epoch, config):
    """beta_y annealing for stage 4: hold at `stage4_beta_y_init` for
    `stage4_beta_y_wait_epochs`, then ramp over `stage4_beta_y_warmup_epochs` to
    the target `stage4_beta_y`. Ramping the rate weight up (instead of holding it
    fixed) lets the encoder build an informative q(y|x) before the I(X;Y|Z) term
    compresses it, which otherwise blows up the y-posterior variance."""
    init = config.stage4_beta_y_init
    final = config.stage4_beta_y
    wait = config.stage4_beta_y_wait_epochs
    warmup = max(config.stage4_beta_y_warmup_epochs, 1)
    if epoch < wait:
        return init
    frac = min((epoch - wait) / warmup, 1.0)
    sched = config.stage4_beta_y_schedule
    if sched == "cosine":
        frac = 0.5 * (1.0 - math.cos(math.pi * frac))  # smooth ease-in
        return init + frac * (final - init)
    if sched == "linear_in_log":  # even progress per decade (needs init > 0)
        lo, hi = math.log(max(init, 1e-12)), math.log(max(final, 1e-12))
        return math.exp(lo + frac * (hi - lo))
    return init + frac * (final - init)  # linear


def build_model(config, device):
    recnet = RecNetConv(config).to(device)
    denoiser_x = DenoiserConv(config).to(device)
    denoiser_y = Denoiser(
        config.latent_dim_y,
        hidden_dim=config.hidden_dim_den_y,
        num_blocks=config.num_blocks_den_y,
        time_dim=config.time_dim_den_y,
        activation=config.activation_den_y,
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


def _apply_resume_state(optimizer, resume_state, stage_label):
    """Continue training in-place: load optimizer momentum and return (start_epoch, step).

    `resume_state` is the dict returned by load_stage_checkpoint when its stage
    matches the stage about to run. Older checkpoints saved without optimizer/step
    (or resume_state=None) fall back to a cold optimizer starting at epoch/step 0.
    """
    if resume_state is None:
        return 0, 0

    start_epoch = resume_state.get("epoch") or 0
    step = resume_state.get("step") or 0
    optimizer_sd = resume_state.get("optimizer_state_dict")
    if optimizer_sd is not None:
        optimizer.load_state_dict(optimizer_sd)
        print(f"[{stage_label}] resumed optimizer state, epoch {start_epoch}, step {step}")
    else:
        print(f"[{stage_label}] resuming weights only (no saved optimizer state), "
              f"epoch {start_epoch}, step {step} — optimizer starts cold")
    return start_epoch, step


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

    # sample y from clean x (canonical source for the running y-scale)
    mu_y_from_x, logvar_y_from_x = hsami.recnet.x_to_y(clean_x, update_stats=True)
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

def train_stage1(hsami, loader, config, device, run_dir, resume_state=None):
    """Stage 1: unconditional DDPM on x, trains denoiser_x only."""
    optimizer = torch.optim.Adam(
        hsami.denoiser_x.parameters(), lr=config.stage1_den_x_lr_init,
    )
    start_epoch, step = _apply_resume_state(optimizer, resume_state, "stage1")

    print(f"=== Stage 1: denoiser_x unconditional ({config.stage1_epochs} epochs) ===")
    if start_epoch >= config.stage1_epochs:
        print(f"  already at epoch {start_epoch} >= stage1_epochs, nothing to do")
        return
    log_every = 20
    print_flag = False

    for epoch in range(start_epoch, config.stage1_epochs):
        if epoch % log_every == 0:
            print_flag = True
        lr = get_lr(epoch, config.stage1_den_x_warmup_epochs, config.stage1_epochs,
                    config.stage1_den_x_lr_init, config.stage1_den_x_lr_final)
        optimizer.param_groups[0]["lr"] = lr

        for x in loader:
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
            save_checkpoint(hsami, run_dir, stage=1, epoch=epoch + 1, optimizer=optimizer, step=step)

    save_checkpoint(hsami, run_dir, stage=1, epoch=config.stage1_epochs, optimizer=optimizer, step=step)
    print("=== Stage 1 complete ===")


def train_stage2(hsami, loader, config, device, run_dir, resume_state=None):
    """Stage 2: denoiser_x + x→y encoder trained jointly via guided x MSE.

    The encoder learns y representations that improve x denoiser guidance.
    denoiser_x continues training; stage 1 weights are NOT frozen.
    """
    enc_xy = _enc_xy_params(hsami)
    optimizer = torch.optim.Adam([
        {"params": hsami.denoiser_x.parameters(), "lr": config.stage2_den_x_lr},
        {"params": enc_xy, "lr": config.stage2_enc_xy_lr},
    ])
    start_epoch, step = _apply_resume_state(optimizer, resume_state, "stage2")

    print(f"=== Stage 2: denoiser_x + x→y encoder ({config.stage2_epochs} epochs) ===")
    if start_epoch >= config.stage2_epochs:
        print(f"  already at epoch {start_epoch} >= stage2_epochs, nothing to do")
        return
    log_every = 20
    print_flag = False

    for epoch in range(start_epoch, config.stage2_epochs):
        if epoch % log_every == 0:
            print_flag = True

        for x in loader:
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
            save_checkpoint(hsami, run_dir, stage=2, epoch=epoch + 1, optimizer=optimizer, step=step)

    save_checkpoint(hsami, run_dir, stage=2, epoch=config.stage2_epochs, optimizer=optimizer, step=step)
    print("=== Stage 2 complete ===")


def train_stage3(hsami, loader, config, device, run_dir, resume_state=None):
    """Stage 3: unconditional denoiser_y on y representations from the encoder.

    y samples are drawn from the (fixed) encoder under no_grad — the denoiser_y
    loss (denoiser_y.compute_loss) detaches its input internally, so no gradients
    reach the encoder or denoiser_x. Those components remain requires_grad=True
    (unfrozen) but are not updated in this stage.
    """
    optimizer = torch.optim.Adam(
        hsami.denoiser_y.parameters(), lr=config.stage3_den_y_lr_init,
    )
    start_epoch, step = _apply_resume_state(optimizer, resume_state, "stage3")

    print(f"=== Stage 3: denoiser_y unconditional ({config.stage3_epochs} epochs) ===")
    if start_epoch >= config.stage3_epochs:
        print(f"  already at epoch {start_epoch} >= stage3_epochs, nothing to do")
        return
    log_every = 20
    print_flag = False

    for epoch in range(start_epoch, config.stage3_epochs):
        if epoch % log_every == 0:
            print_flag = True
        lr = get_lr(epoch, config.stage3_den_y_warmup_epochs, config.stage3_epochs,
                    config.stage3_den_y_lr_init, config.stage3_den_y_lr_final)
        optimizer.param_groups[0]["lr"] = lr

        for x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                mu_y, logvar_y = hsami.recnet.x_to_y(x, update_stats=True)
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
            save_checkpoint(hsami, run_dir, stage=3, epoch=epoch + 1, optimizer=optimizer, step=step)

    save_checkpoint(hsami, run_dir, stage=3, epoch=config.stage3_epochs, optimizer=optimizer, step=step)
    print("=== Stage 3 complete ===")


@torch.no_grad()
def _linear_r2(feats: np.ndarray, targets: np.ndarray) -> float:
    """R^2 of the best least-squares linear map feats -> targets (intercept
    included), averaged over target columns. Measures how much of the ground-truth
    factor is linearly decodable from the latent."""
    A = np.concatenate([feats, np.ones((feats.shape[0], 1))], axis=1)
    T = targets.reshape(feats.shape[0], -1)
    coef, *_ = np.linalg.lstsq(A, T, rcond=None)
    pred = A @ coef
    ss_res = ((T - pred) ** 2).sum(0)
    ss_tot = ((T - T.mean(0)) ** 2).sum(0) + 1e-12
    return float((1.0 - ss_res / ss_tot).mean())


@torch.no_grad()
def probe_latents(hsami, dataset, device, max_samples: int = 2000, batch: int = 512):
    """Linear-probe the encoder latents against the dataset's stored ground-truth
    generative factors (z=d gauge, per-disk intensities I, vertical positions cy).

    Tells you whether y/z are *informative* even when mse_x is insensitive to them
    (intensity/distance are low-energy in pixel space here). High R^2 => the latent
    captures the factor; a linear map suffices because the generative model is
    (conditionally) Gaussian-linear. Returns a dict of R^2 values + posterior stds.
    """
    was_training = hsami.training
    hsami.eval()  # freezes the running y-scale (no update in eval) during probing

    n = min(max_samples, dataset.targets.shape[0])
    mu_y_list, mu_z_list, sy_list, sz_list = [], [], [], []
    for i in range(0, n, batch):
        x = dataset.targets[i:min(i + batch, n)].to(device)
        x = normalize_to_pm1(x)  # match the training input convention ([-1, 1])
        mu_y, logvar_y, mu_z, logvar_z = hsami.recnet.x_to_y_to_z(x)
        mu_y_list.append(mu_y.cpu().numpy()); mu_z_list.append(mu_z.cpu().numpy())
        sy_list.append((0.5 * logvar_y).exp().mean().item())
        sz_list.append((0.5 * logvar_z).exp().mean().item())
    Y = np.concatenate(mu_y_list, 0); Z = np.concatenate(mu_z_list, 0)

    out: dict = {}
    if hasattr(dataset, "data_d"):
        d = dataset.data_d[:n].numpy()
        out["z->d"] = _linear_r2(Z, d)
        out["yz->d"] = _linear_r2(np.concatenate([Y, Z], 1), d)  # is d leaking into y?
    if hasattr(dataset, "data_I"):
        out["y->I"] = _linear_r2(Y, dataset.data_I[:n].numpy())
    if hasattr(dataset, "data_cy"):
        out["y->cy"] = _linear_r2(Y, dataset.data_cy[:n].numpy())
    out["std_y"] = float(np.mean(sy_list))
    out["std_z"] = float(np.mean(sz_list))
    if hasattr(dataset, "gauge_mi_nats"):
        out["I(z;x)nats"] = float(dataset.gauge_mi_nats)

    if was_training:
        hsami.train()
    return out


def train_stage4(hsami, loader, config, device, run_dir, resume_state=None):
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
    start_epoch, step = _apply_resume_state(optimizer, resume_state, "stage4")

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
    if start_epoch >= config.stage4_epochs:
        print(f"  already at epoch {start_epoch} >= stage4_epochs, nothing to do")
        return
    log_every = 20
    print_flag = False

    for epoch in range(start_epoch, config.stage4_epochs):
        if epoch % log_every == 0:
            print_flag = True

        beta_y = get_beta_y(epoch, config)

        for x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            t_y_max = get_t_y_max(
                step, t_y_hard_max, config.num_timesteps_y,
                config.stage4_t_y_curriculum_steps,
            )
            total_loss, mse_x, mse_y, rate_y_loss, rate_z_loss, debug_norms = hsami.compute_loss(
                x,
                beta_y=beta_y,
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
                    f"beta_y {beta_y:.2e} | "
                    f"t_y_max {t_y_max}/{config.num_timesteps_y - 1}"
                )
                if debug_norms:
                    print("  norms: " + " | ".join(f"{k} {v:.3e}" for k, v in debug_norms.items()))
                print_flag = False
            step += 1

        # periodic latent probe: is y/z actually informative (independent of mse_x)?
        if epoch % log_every == 0 and hasattr(loader.dataset, "targets"):
            probe = probe_latents(hsami, loader.dataset, device)
            print("  probe R^2: " + " | ".join(f"{k} {v:.3f}" for k, v in probe.items()))

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

    if config.dataset_name == "hierarchical_disks":
        dataset = HierarchicalDiskDataset(
            num_samples=config.num_samples,
            grid_size=config.grid_size,
            foreground = config.foreground,
            background = config.background,
            outer_radius = config.outer_radius,
            transition_width = config.transition_width,
            dataset_seed = config.dataset_seed,
            a = config.a,
            b = config.b,
            sigma = config.sigma,
            n_std = config.n_std,
        )
        checkpoint_name = "hsami_checkpoints_hdisks"
    elif config.dataset_name == "hierarchical_two_disks":
        dataset = HierarchicalTwoDiskIntensityDataset(
            num_samples = config.num_samples,
            grid_size = config.grid_size,
            background = config.background,
            mu_I = config.mu_I,
            sigma_b = config.sigma_b,
            sigma_y = config.sigma_y,
            I_min = config.I_min,
            I_max = config.I_max,
            outer_radius = config.outer_radius,
            transition_width = config.transition_width,
            dataset_seed = config.dataset_seed,
        )
        checkpoint_name = "hsami_checkpoints_h2disks"
    elif config.dataset_name == "h2disks":
        dataset = HierarchicalTwoDiskDistanceIntensityDataset(
            num_samples = config.num_samples,
            grid_size = config.grid_size,
            mu_I = config.mu_I,
            background = config.background,
            sigma_d = config.sigma_d,
            sigma_eps = config.sigma_eps,
            intensity_gain = config.intensity_gain,
            sigma_eta = config.sigma_eta,
            I_min = config.I_min,
            I_max = config.I_max,
            outer_radius = config.outer_radius,
            transition_width = config.transition_width,
            dataset_seed = config.dataset_seed,
            transform = normalize_to_pm1,
        )
        checkpoint_name = "hsami_checkpoints_h2disks"
    else:
        raise ValueError(f"unknown dataset_name: {config.dataset_name}")
    
    loader = DataLoader(
        dataset,
        batch_size=config.train_batch_size_per_gpu,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    hsami = build_model(config, device)

    checkpoint_dir = os.path.join(project_root, "c_training", checkpoint_name)
    run_dir = os.path.join(
        checkpoint_dir, "sequential_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)
    save_config(config, run_dir)
    save_sequential_config(config, run_dir)

    tee = Tee(os.path.join(run_dir, "train.out"))
    sys.stdout = tee
    num_params = sum(p.numel() for p in hsami.parameters() if p.requires_grad)
    print(f"parameters: {num_params:,}")
    print(f"run dir: {run_dir}")

    stages = sorted(config.stages)
    first_stage = stages[0]
    resume_state = None  # optimizer/step continuation, only set if resuming into first_stage itself
    if first_stage > 1:
        resume_checkpoints = config.resume_checkpoints
        if not resume_checkpoints:
            if not config.resume_from_run:
                raise ValueError(
                    f"stages starts at {first_stage} but neither resume_checkpoints "
                    f"nor resume_from_run is set"
                )
            resume_checkpoints = [(first_stage - 1, config.resume_from_run)]
        for stage, run_name in resume_checkpoints:
            info = load_stage_checkpoint(hsami, checkpoint_dir, run_name, stage, device)
            if stage == first_stage:
                resume_state = info

    stage_fns = {
        1: train_stage1,
        2: train_stage2,
        3: train_stage3,
        4: train_stage4,
    }
    for stage in stages:
        stage_fns[stage](
            hsami, loader, config, device, run_dir,
            resume_state=(resume_state if stage == first_stage else None),
        )

    tee.close()


if __name__ == "__main__":
    train()
