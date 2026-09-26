"""Rate and distortion of trained SAMI models on dSprites.

Image version of d_analysis/utils/sami_simple.get_rd: for each image x0 we draw a noise level t
(uniform over 1..T-1, or fixed with --t), make x_t, and measure

  distortion  ||eps - eps_hat(x_t, t | z)||^2, the (unweighted) guided noise-prediction error,
              with eps_hat = denoiser(x_t, t) - sqrt(gamma_t) * grad_{x_t} log q(z | x_t),  z ~ q(z | x0)
  rate (norm) ||grad_{x_t} log q(z | x_t)||^2, as in get_rd(rate_eval_method='norm')
  rate (kl)   KL(q(z | x0) || N(0, I)),         as in get_rd(rate_eval_method='kl')

Values are reported per image (summed over the 64x64 pixels, the "sum" convention in
SAMI.compute_loss) and per pixel (divided by 4096, what wandb logs for reduction="mean").
Both are averages over --num-images random dSprites images x --num-draws noise draws, +- standard error.

Usage:
    python d_analysis/rd_dsprites.py --model-num 11
    python d_analysis/rd_dsprites.py --model-num 2 5 9 11 15 --num-images 20000
    python d_analysis/rd_dsprites.py --model-num 11 --t 50 --epoch 1199
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from d_analysis.probe_disentanglement import DSPRITES_DIR, find_checkpoint, get_model_builders  # noqa: E402


# ---------------------------------------------------------------------------- #
#                                 model loading                                #
# ---------------------------------------------------------------------------- #
def load_sami(model_type, project, model_num, epoch="last", device="cpu"):
    """Rebuild the full SAMI model (denoiser + infnet) from a local Lightning checkpoint, frozen and in eval mode."""
    config_cls, init_model = get_model_builders(model_type)
    ckpt_path = find_checkpoint(project, model_num, epoch)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = config_cls.from_dict(ckpt["hyper_parameters"])

    model = init_model(config)
    state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state_dict)
    model = model.to(device).eval().requires_grad_(False)
    print(f"loaded {model_type} model from {ckpt_path}")
    return model, config, ckpt_path


def load_dsprites_subset(num_images, seed=0, data_dir=DSPRITES_DIR):
    """`num_images` random dSprites images as a (N, 1, 64, 64) float tensor in [-1, 1]."""
    images = np.load(f"{data_dir}/dataset_images.npy", mmap_mode="r")
    idx = np.sort(np.random.default_rng(seed).choice(len(images), num_images, replace=False))
    x = torch.from_numpy(np.array(images[idx])).float().unsqueeze(1)
    return x * 2 - 1, idx  # [0, 1] -> [-1, 1], same as the training transform Normalize(0.5, 0.5)


# ---------------------------------------------------------------------------- #
#                                rate/distortion                               #
# ---------------------------------------------------------------------------- #
def rd_batch(sami, clean_x, t=None):
    """Per-image distortion, norm rate and KL rate (each shape (B,)) for one batch of clean images.

    Mirrors get_rd_single_gamma in d_analysis/utils/sami_simple.py, but draws one t per image
    rather than one per batch (same expectation, lower variance).
    """
    if sami.parameterization != "noise":
        raise NotImplementedError(f"only the 'noise' parameterization is supported, got {sami.parameterization}")
    B = clean_x.shape[0]
    if t is None:
        timestep = torch.randint(1, sami.num_timesteps, (B,), device=clean_x.device)
    else:
        assert 0 < t < sami.num_timesteps, f"t must be in [1, {sami.num_timesteps - 1}]"
        timestep = torch.full((B,), int(t), dtype=torch.long, device=clean_x.device)

    noisy_x, noise = sami.make_noisy(clean_x, timestep)
    noisy_x = noisy_x.requires_grad_(True)
    gamma = sami.extract(sami.one_minus_alpha_bars, timestep, clean_x.shape)

    # z ~ q(z | x0) from the clean view, and its score under the noisy view
    mu, logvar = sami.encode(sami.infnet, clean_x, None)
    z_sample = sami.infnet.sample(mu, logvar)
    mu_t, logvar_t = sami.encode(sami.infnet, noisy_x, timestep)
    log_posterior = sami.compute_log_posterior(mu_t, logvar_t, z_sample)
    score = torch.autograd.grad(log_posterior.sum(), noisy_x)[0]

    with torch.no_grad():
        pred_noise_guided = sami.denoiser(noisy_x, timestep) - gamma.sqrt() * score
        distortion = (noise - pred_noise_guided).square().flatten(1).sum(dim=1)
        rate_norm = score.flatten(1).square().sum(dim=1)
        rate_kl = 0.5 * (logvar.exp() + mu.square() - 1 - logvar).sum(dim=1)
    return distortion, rate_norm, rate_kl, timestep


def get_rd_dsprites(sami, images, batch_size=256, num_draws=1, t=None, device="cpu"):
    """Per-sample distortion, rate and timestep over all images x num_draws noise draws (numpy arrays)."""
    out = {"distortion": [], "rate_norm": [], "rate_kl": [], "timestep": []}
    num_batches = num_draws * ((len(images) + batch_size - 1) // batch_size)
    with tqdm(total=num_batches, desc="rate/distortion") as pbar:
        for _ in range(num_draws):
            for start in range(0, len(images), batch_size):
                clean_x = images[start : start + batch_size].to(device)
                for key, val in zip(out, rd_batch(sami, clean_x, t)):
                    out[key].append(val.cpu())
                pbar.update()
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def summarize(samples, data_dim):
    """Mean and standard error, per image and per pixel."""
    summary = {"per_image": {}, "per_pixel": {}}
    for key in ["distortion", "rate_norm", "rate_kl"]:
        vals = samples[key].astype(np.float64)
        mean, sem = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
        summary["per_image"][key] = {"mean": float(mean), "sem": float(sem)}
        summary["per_pixel"][key] = {"mean": float(mean / data_dim), "sem": float(sem / data_dim)}
    return summary


# ---------------------------------------------------------------------------- #
#                                     main                                     #
# ---------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-type", choices=["sami"], default="sami")
    p.add_argument("--project", default="imae_dsprites", help="folder under c_training/lightning_checkpoints")
    p.add_argument("--model-num", type=int, nargs="+", default=[11], help="one or more run numbers")
    p.add_argument("--epoch", default="last", help="'last' or an epoch number, e.g. 1199")
    p.add_argument("--num-images", type=int, default=10000, help="random dSprites images to evaluate")
    p.add_argument("--num-draws", type=int, default=1, help="noise draws (t, eps, z) per image")
    p.add_argument("--t", type=int, default=None, help="fixed timestep; default samples t ~ U{1..T-1}")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=f"{project_dir}/d_analysis/metrics")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    images, idx = load_dsprites_subset(args.num_images, args.seed)
    data_dim = images[0].numel()

    rows = []
    for model_num in args.model_num:
        try:
            sami, config, ckpt_path = load_sami(args.model_type, args.project, model_num, args.epoch, args.device)
        except FileNotFoundError as e:
            print(f"skipping model {model_num}: {e}")
            continue

        torch.manual_seed(args.seed)
        samples = get_rd_dsprites(sami, images, args.batch_size, args.num_draws, args.t, args.device)
        summary = summarize(samples, data_dim)
        rows.append((model_num, config, summary))

        tag = f"{args.project}_{model_num}_{args.epoch}" + (f"_t{args.t}" if args.t is not None else "")
        output = {
            "args": {**vars(args), "model_num": model_num},
            "checkpoint": ckpt_path,
            "beta_final": config.beta_final,
            "rate_type": config.rate_type,
            "num_samples": len(samples["distortion"]),
            **summary,
        }
        os.makedirs(args.out_dir, exist_ok=True)
        out_json = f"{args.out_dir}/rd_{tag}.json"
        with open(out_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"saved results to {out_json}")

    print(f"\n=== rate / distortion on dSprites ({args.num_images} images x {args.num_draws} draws, "
          f"t = {args.t if args.t is not None else 'U{1..T-1}'}; mean +- s.e.) ===")
    for unit in ["per_image", "per_pixel"]:
        print(f"\n[{unit.replace('_', ' ')}]")
        print(f"{'model':>6}{'beta':>10}{'rate_type':>10}{'distortion':>24}{'rate (norm)':>24}{'rate (kl)':>24}")
        for model_num, config, summary in rows:
            cells = "".join(f"{s['mean']:>14.4g} +- {s['sem']:<6.2g}" for s in summary[unit].values())
            print(f"{model_num:>6}{config.beta_final:>10.0e}{config.rate_type:>10}{cells}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
