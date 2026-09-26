"""Disentanglement evaluation of a trained encoder on dSprites: probes, FactorVAE score and MIG.

Encodes every clean dSprites image once (cached to d_analysis/metrics/codes/), takes the posterior
means mu(x), and computes:

  - probes: small probes (logistic/ridge regression and/or an MLP) that predict each factor from mu
  - factorvae: FactorVAE score (Kim & Mnih, 2018), see d_analysis/metrics/factor_vae.py
  - mig: Mutual Information Gap (Chen et al., 2018), see d_analysis/metrics/mig.py

FactorVAE and MIG follow disentanglement_lib (same defaults, raw factor classes including all 40
orientation values) so the numbers are comparable to published dSprites results; they are repeated
--num-repeats times with different seeds and reported as mean +- std.

The probes fit on a held-out split and give two views:

1. Full-latent probes: can each factor be read out of the whole code at all?
   (informativeness / explicitness; linear vs MLP tells you how nonlinearly it's stored)
2. Per-dimension probes: fit one probe per (latent dim, factor) using that dim alone.
   A disentangled code has one bright cell per factor row, and each latent dim
   serves at most one factor. The "gap" (best dim minus second-best dim, as in SAP)
   summarises this per factor.

Factors and targets (dSprites latents_values columns: color, shape, scale, orientation, posX, posY):
    shape        3-way classification (square, ellipse, heart); score = accuracy
    scale        regression; score = R^2
    orientation  regression onto (cos n*theta, sin n*theta), where n is the shape's
                 rotational symmetry (square 4, ellipse 2, heart 1), because a square
                 at theta and theta + pi/2 is the same image; score = R^2
    pos_x, pos_y regression; score = R^2
Classification accuracy is also reported chance-normalized, (acc - 1/3) / (1 - 1/3),
so every score lives on the same 0 (chance) to 1 (perfect) scale.

Usage:
    python d_analysis/probe_disentanglement.py --model-type sami --project imae_dsprites --model-num 11
    python d_analysis/probe_disentanglement.py --model-num 11 --epoch 399 --metrics factorvae mig
    python d_analysis/probe_disentanglement.py --model-num 11 --probe linear --per-dim-probe linear
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tqdm import tqdm  # noqa: E402

from b_models.sami.sami_module import SAMI  # noqa: E402
from d_analysis.metrics.factor_vae import compute_factor_vae  # noqa: E402
from d_analysis.metrics.mig import compute_mig  # noqa: E402

DSPRITES_DIR = f"{project_dir}/a_datasets/dsprites"
CHECKPOINT_DIR = f"{project_dir}/c_training/lightning_checkpoints"

FACTORS = ["shape", "scale", "orientation", "pos_x", "pos_y"]
SHAPE_NAMES = {0: "square", 1: "ellipse", 2: "heart"}
SHAPE_SYMMETRY = np.array([4, 2, 1])  # rotational symmetry order per shape class
NUM_SHAPES = 3
# dSprites is stored as a full grid over (color, shape, scale, orientation, posX, posY);
# color is constant, so image index = factor_classes @ FACTOR_STRIDES over the five factors above
FACTOR_SIZES = np.array([3, 6, 40, 32, 32])
FACTOR_STRIDES = np.array([40 * 32 * 32 * 6, 40 * 32 * 32, 32 * 32, 32, 1])


# ---------------------------------------------------------------------------- #
#                                 model loading                                #
# ---------------------------------------------------------------------------- #
def get_model_builders(model_type):
    """(config class, model init fn) for each supported model type. Imported lazily."""
    if model_type == "sami":
        from b_models.configs.sami_config_disent import Config
        from utils.model_init import init_sami_model

        return Config, init_sami_model
    # NOTE: DiVA checkpoints (diva_convnet_dsprites) don't build with the current UNet/ConvNetComplex
    # (their config lacks norm, norm_rec, ...). For other models, pass their codes to run_probes() directly.
    raise ValueError(f"Unknown model_type: {model_type}. Supported: 'sami'.")


def find_checkpoint(project, model_num, epoch="last", checkpoint_dir=CHECKPOINT_DIR):
    """Path to a local Lightning checkpoint, e.g. .../imae_dsprites/11-astral-sky-11-xxxx/last.ckpt.

    Matches the run folder on "{model_num}-" so that model 1 does not pick up run 10/11.
    """
    project_path = f"{checkpoint_dir}/{project}"
    run_dirs = [d for d in os.listdir(project_path) if d.startswith(f"{model_num}-")]
    if len(run_dirs) != 1:
        raise FileNotFoundError(f"Expected one run folder for model {model_num} in {project_path}, found {run_dirs}")
    ckpt_name = "last.ckpt" if epoch in (None, "last", "latest") else f"epoch={int(epoch):04d}.ckpt"
    ckpt_path = f"{project_path}/{run_dirs[0]}/{ckpt_name}"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} does not exist")
    return ckpt_path


def load_encoder(model_type, project, model_num, epoch="last", device="cpu"):
    """Rebuild the model from the config saved in the checkpoint and return its infnet (eval mode)."""
    config_cls, init_model = get_model_builders(model_type)
    ckpt_path = find_checkpoint(project, model_num, epoch)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = config_cls.from_dict(ckpt["hyper_parameters"])

    model = init_model(config)
    state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state_dict)
    encoder = model.infnet.to(device).eval()
    print(f"loaded {model_type} encoder from {ckpt_path}")
    return encoder, config, ckpt_path


# ---------------------------------------------------------------------------- #
#                               data and encoding                              #
# ---------------------------------------------------------------------------- #
def load_dsprites_labels(data_dir=DSPRITES_DIR):
    """Factor values (N x 6, latents_values) and integer factor classes (N x 5) of every dSprites image."""
    labels = np.load(f"{data_dir}/dataset_labels.npy")
    factor_classes = (np.arange(len(labels))[:, None] // FACTOR_STRIDES) % FACTOR_SIZES
    return labels, factor_classes


@torch.no_grad()
def encode_clean(encoder, images, batch_size=512, device="cpu", progress=False):
    """Posterior means and log-variances of clean images (timestep 0 for time-conditioned encoders).

    `images` can be a (memory-mapped) uint8 array of shape (N, 64, 64) with values in {0, 1}.
    """
    mus, logvars = [], []
    for start in tqdm(range(0, len(images), batch_size), desc="encoding", disable=not progress):
        x = torch.from_numpy(np.array(images[start : start + batch_size])).float().unsqueeze(1).to(device)
        x = x * 2 - 1  # [0, 1] -> [-1, 1], same as the training transform Normalize(0.5, 0.5)
        mu, logvar = SAMI.encode(encoder, x)
        mus.append(mu.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
    return np.concatenate(mus), np.concatenate(logvars)


def get_dsprites_codes(encoder, ckpt_path, cache_path, batch_size=512, device="cpu", recompute=False):
    """Encode all 737280 dSprites images once, caching (mu, logvar) next to the metrics.

    The cache is keyed on the checkpoint's modification time, so an updated last.ckpt is re-encoded.
    """
    ckpt_mtime = os.path.getmtime(ckpt_path)
    if not recompute and os.path.exists(cache_path):
        cache = np.load(cache_path)
        if str(cache["ckpt_path"]) == ckpt_path and float(cache["ckpt_mtime"]) == ckpt_mtime:
            print(f"loaded cached codes from {cache_path}")
            return cache["mu"], cache["logvar"]
        print(f"cache {cache_path} is from a different checkpoint, re-encoding")

    images = np.load(f"{DSPRITES_DIR}/dataset_images.npy", mmap_mode="r")
    mu, logvar = encode_clean(encoder, images, batch_size, device, progress=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, mu=mu, logvar=logvar, ckpt_path=ckpt_path, ckpt_mtime=ckpt_mtime)
    print(f"cached codes to {cache_path}")
    return mu, logvar


def make_targets(labels):
    """Probe targets from dSprites latents_values. Returns {factor: (y, task)}."""
    shape = labels[:, 1].astype(int) - 1  # 1,2,3 -> 0,1,2
    theta = labels[:, 3]
    n = SHAPE_SYMMETRY[shape]
    return {
        "shape": (shape, "classification"),
        "scale": (labels[:, 2], "regression"),
        "orientation": (np.stack([np.cos(n * theta), np.sin(n * theta)], axis=1), "regression"),
        "pos_x": (labels[:, 4], "regression"),
        "pos_y": (labels[:, 5], "regression"),
    }


# ---------------------------------------------------------------------------- #
#                                    probes                                    #
# ---------------------------------------------------------------------------- #
def make_probe(kind, task, seed=0, hidden=(64, 64)):
    if kind == "linear":
        head = LogisticRegression(max_iter=2000) if task == "classification" else Ridge(alpha=1.0)
    elif kind == "mlp":
        mlp_cls = MLPClassifier if task == "classification" else MLPRegressor
        head = mlp_cls(hidden_layer_sizes=hidden, max_iter=500, early_stopping=True, random_state=seed)
    else:
        raise ValueError(f"Unknown probe kind: {kind}")
    return make_pipeline(StandardScaler(), head)


def score_probe(probe, x, y, task):
    """Score in [~0, 1]: R^2 for regression, chance-normalized accuracy for classification."""
    pred = probe.predict(x)
    if task == "classification":
        acc = accuracy_score(y, pred)
        chance = 1 / NUM_SHAPES
        return (acc - chance) / (1 - chance), {"accuracy": float(acc)}
    return r2_score(y, pred), {}


def fit_and_score(kind, x_train, y_train, x_test, y_test, task, seed=0, hidden=(64, 64)):
    probe = make_probe(kind, task, seed, hidden)
    probe.fit(x_train, y_train)
    test_score, extras = score_probe(probe, x_test, y_test, task)
    train_score, _ = score_probe(probe, x_train, y_train, task)
    return probe, {"test": float(test_score), "train": float(train_score), **extras}


def orientation_breakdown(probe, x_test, y_test, shape_test):
    """Per-shape R^2 and mean angular error (degrees, in the shape's own symmetry period)."""
    pred = probe.predict(x_test)
    out = {}
    for s, name in SHAPE_NAMES.items():
        m = shape_test == s
        n = SHAPE_SYMMETRY[s]
        dphi = np.angle(np.exp(1j * (np.arctan2(pred[m, 1], pred[m, 0]) - np.arctan2(y_test[m, 1], y_test[m, 0]))))
        out[name] = {
            "r2": float(r2_score(y_test[m], pred[m])),
            "mean_abs_angle_error_deg": float(np.degrees(np.abs(dphi)).mean() / n),
            "period_deg": 360 / n,
        }
    return out


def run_probes(mu_train, labels_train, mu_test, labels_test, probes=("linear", "mlp"),
               per_dim_probe="linear", seed=0, hidden=(64, 64)):
    """Fit all probes. Usable from a notebook with any (N x z_dim) codes and dSprites latents_values."""
    targets_train, targets_test = make_targets(labels_train), make_targets(labels_test)
    shape_test = targets_test["shape"][0]
    results = {"full_latent": {}, "per_dim": {}}

    # 1) full-latent probes
    for kind in probes:
        results["full_latent"][kind] = {}
        for factor in FACTORS:
            (y_tr, task), (y_te, _) = targets_train[factor], targets_test[factor]
            probe, res = fit_and_score(kind, mu_train, y_tr, mu_test, y_te, task, seed, hidden)
            if factor == "orientation":
                res["per_shape"] = orientation_breakdown(probe, mu_test, y_te, shape_test)
            results["full_latent"][kind][factor] = res

    # 2) per-dimension probes: score matrix of shape (num_factors, z_dim)
    if per_dim_probe is not None:
        z_dim = mu_train.shape[1]
        matrix = np.zeros((len(FACTORS), z_dim))
        for i, factor in enumerate(FACTORS):
            (y_tr, task), (y_te, _) = targets_train[factor], targets_test[factor]
            for j in range(z_dim):
                _, res = fit_and_score(per_dim_probe, mu_train[:, [j]], y_tr, mu_test[:, [j]], y_te, task, seed, hidden)
                matrix[i, j] = res["test"]
        clipped = np.clip(matrix, 0, None)
        top2 = np.sort(clipped, axis=1)[:, ::-1][:, :2]
        results["per_dim"] = {
            "probe": per_dim_probe,
            "factors": FACTORS,
            "matrix": matrix.tolist(),
            "best_dim": {f: int(np.argmax(clipped[i])) for i, f in enumerate(FACTORS)},
            "gap": {f: float(top2[i, 0] - top2[i, 1]) for i, f in enumerate(FACTORS)},
            "sap_score": float((top2[:, 0] - top2[:, 1]).mean()),
            # does each latent dim serve a single factor? 1 = one factor, 0 = spread evenly
            "dim_exclusivity": _dim_exclusivity(clipped).tolist(),
        }
    return results


def _dim_exclusivity(matrix):
    """1 - normalized entropy of each column (latent dim) over factors, like DCI disentanglement per code."""
    p = matrix / np.clip(matrix.sum(axis=0, keepdims=True), 1e-12, None)
    entropy = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=0) / np.log(matrix.shape[0])
    return np.where(matrix.sum(axis=0) > 1e-6, 1 - entropy, 0.0)


# ---------------------------------------------------------------------------- #
#                               FactorVAE and MIG                              #
# ---------------------------------------------------------------------------- #
def run_factor_vae(mu_all, num_repeats=5, seed=0):
    """FactorVAE score on codes for the full dSprites grid, repeated over seeds."""
    runs = [compute_factor_vae(mu_all, FACTOR_SIZES, FACTOR_STRIDES, np.random.RandomState(seed + r))
            for r in range(num_repeats)]
    evals = np.array([r["eval_accuracy"] for r in runs])
    return {
        "eval_accuracy_mean": float(evals.mean()),
        "eval_accuracy_std": float(evals.std()),
        "train_accuracy_mean": float(np.mean([r["train_accuracy"] for r in runs])),
        "runs": runs,
    }


def run_mig(mu_all, factor_classes, num_repeats=5, seed=0, num_points=10000, num_bins=20):
    """MIG on `num_points` random images (disentanglement_lib default: 10000), repeated over seeds."""
    runs = []
    for r in range(num_repeats):
        idx = np.random.RandomState(seed + r).choice(len(mu_all), num_points, replace=False)
        runs.append(compute_mig(mu_all[idx], factor_classes[idx], num_bins))
    migs = np.array([r["mig"] for r in runs])
    return {
        "mig_mean": float(migs.mean()),
        "mig_std": float(migs.std()),
        "mig_per_factor_mean": np.mean([r["mig_per_factor"] for r in runs], axis=0).tolist(),
        "mi_matrix_mean": np.mean([r["mi_matrix"] for r in runs], axis=0).tolist(),
        "num_points": num_points,
        "num_bins": num_bins,
        "runs": runs,
    }


# ---------------------------------------------------------------------------- #
#                                   reporting                                  #
# ---------------------------------------------------------------------------- #
def latent_stats(mu, logvar):
    """Per-dim spread of the means across data and average posterior variance (collapsed dims: std(mu) ~ 0)."""
    return {"std_mu": mu.std(axis=0).tolist(), "mean_posterior_var": np.exp(logvar).mean(axis=0).tolist()}


def print_latent_stats(stats):
    print("\n=== latent dims (all images) ===")
    print("dim            " + "".join(f"{j:>8d}" for j in range(len(stats["std_mu"]))))
    print("std(mu)        " + "".join(f"{v:8.3f}" for v in stats["std_mu"]))
    print("mean post. var " + "".join(f"{v:8.1e}" for v in stats["mean_posterior_var"]))


def print_probe_report(results):
    print("\n=== full-latent probes (test; R^2 or chance-normalized accuracy; 0 = chance, 1 = perfect) ===")
    kinds = list(results["full_latent"])
    print(f"{'factor':<12}" + "".join(f"{k:>10}" for k in kinds))
    for factor in FACTORS:
        row = "".join(f"{results['full_latent'][k][factor]['test']:10.3f}" for k in kinds)
        extra = ""
        if factor == "shape":
            extra = "   acc: " + ", ".join(f"{k} {results['full_latent'][k]['shape']['accuracy']:.3f}" for k in kinds)
        print(f"{factor:<12}{row}{extra}")
    for k in kinds:
        per_shape = results["full_latent"][k]["orientation"]["per_shape"]
        desc = ", ".join(
            f"{name} R2={v['r2']:.2f} err={v['mean_abs_angle_error_deg']:.1f}deg/{v['period_deg']:.0f}"
            for name, v in per_shape.items()
        )
        print(f"  orientation [{k}] by shape: {desc}")

    if results["per_dim"]:
        pd = results["per_dim"]
        matrix = np.array(pd["matrix"])
        print(f"\n=== per-dim probes ({pd['probe']}; one latent dim at a time, test score) ===")
        print(f"{'factor':<12}" + "".join(f"{f'z{j}':>7}" for j in range(matrix.shape[1])) + "   best   gap")
        for i, factor in enumerate(FACTORS):
            print(
                f"{factor:<12}" + "".join(f"{v:7.2f}" for v in matrix[i])
                + f"   z{pd['best_dim'][factor]:<4d}{pd['gap'][factor]:6.2f}"
            )
        print(f"{'exclusivity':<12}" + "".join(f"{v:7.2f}" for v in pd["dim_exclusivity"]))
        print(f"SAP-style score (mean gap): {pd['sap_score']:.3f}")


def print_factor_vae_report(fvae):
    runs = fvae["runs"]
    print(f"\n=== FactorVAE score ({len(runs)} repeats) ===")
    print(f"eval accuracy {fvae['eval_accuracy_mean']:.3f} +- {fvae['eval_accuracy_std']:.3f}"
          f"   (train {fvae['train_accuracy_mean']:.3f}; chance {1 / len(FACTORS):.2f})")
    first = runs[0]
    print(f"active dims (var >= 0.05): {first['active_dims']}")
    assignment = ", ".join(f"z{d}->{FACTORS[f]}" for d, f in zip(first["active_dims"], first["dim_to_factor"]))
    print(f"majority-vote assignment (repeat 0): {assignment}")


def print_mig_report(mig):
    print(f"\n=== MIG ({len(mig['runs'])} repeats, {mig['num_points']} points, {mig['num_bins']} bins) ===")
    print(f"MIG {mig['mig_mean']:.3f} +- {mig['mig_std']:.3f}")
    print("per factor: " + ", ".join(f"{f} {v:.3f}" for f, v in zip(FACTORS, mig["mig_per_factor_mean"])))
    mi = np.array(mig["mi_matrix_mean"])  # (num_codes, num_factors)
    print(f"{'MI (nats)':<12}" + "".join(f"{f'z{j}':>7}" for j in range(mi.shape[0])))
    for i, factor in enumerate(FACTORS):
        print(f"{factor:<12}" + "".join(f"{v:7.2f}" for v in mi[:, i]))


def plot_per_dim(results, title, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    pd = results["per_dim"]
    matrix = np.clip(np.array(pd["matrix"]), 0, 1)
    ramp = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ramp)

    n_f, n_z = matrix.shape
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * n_z, 0.9 + 0.55 * n_f))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(n_f):
        for j in range(n_z):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if matrix[i, j] > 0.55 else "#1f1f1f")
    ax.set_xticks(range(n_z), [f"z{j}" for j in range(n_z)])
    ax.set_yticks(range(n_f), pd["factors"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_z), minor=True)
    ax.set_yticks(np.arange(-0.5, n_f), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=10, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"{pd['probe']} probe score\n(R$^2$ / chance-norm. acc.)", fontsize=8)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure to {out_path}")


# ---------------------------------------------------------------------------- #
#                                     main                                     #
# ---------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-type", choices=["sami"], default="sami")
    p.add_argument("--project", default="imae_dsprites", help="folder under c_training/lightning_checkpoints")
    p.add_argument("--model-num", type=int, default=11)
    p.add_argument("--epoch", default="last", help="'last' or an epoch number, e.g. 599")
    p.add_argument("--metrics", nargs="+", choices=["probes", "factorvae", "mig"], default=["probes", "factorvae", "mig"])
    p.add_argument("--num-repeats", type=int, default=5, help="seeds for FactorVAE and MIG")
    p.add_argument("--num-train", type=int, default=30000, help="probe training images")
    p.add_argument("--num-test", type=int, default=10000, help="probe test images")
    p.add_argument("--probe", choices=["linear", "mlp", "both"], default="both")
    p.add_argument("--per-dim-probe", choices=["linear", "mlp", "none"], default="mlp")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64], help="MLP probe hidden sizes")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--recompute-codes", action="store_true", help="ignore the cached encoding of dSprites")
    p.add_argument("--out-dir", default=f"{project_dir}/d_analysis/metrics")
    p.add_argument("--fig-dir", default=f"{project_dir}/d_analysis/figures")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    tag = f"{args.project}_{args.model_num}_{args.epoch}"

    encoder, _, ckpt_path = load_encoder(args.model_type, args.project, args.model_num, args.epoch, args.device)
    cache_path = f"{args.out_dir}/codes/{tag}.npz"
    mu, logvar = get_dsprites_codes(encoder, ckpt_path, cache_path, args.batch_size, args.device, args.recompute_codes)
    labels, factor_classes = load_dsprites_labels()
    print(f"codes for {len(mu)} images -> z_dim {mu.shape[1]} ({time.time() - t0:.0f}s)")

    stats = latent_stats(mu, logvar)
    print_latent_stats(stats)
    output = {"args": vars(args), "checkpoint": ckpt_path, "latent_stats": stats}

    if "probes" in args.metrics:
        idx = np.random.default_rng(args.seed).choice(len(mu), args.num_train + args.num_test, replace=False)
        train, test = idx[: args.num_train], idx[args.num_train :]
        probes = ("linear", "mlp") if args.probe == "both" else (args.probe,)
        per_dim_probe = None if args.per_dim_probe == "none" else args.per_dim_probe
        results = run_probes(mu[train], labels[train], mu[test], labels[test],
                             probes, per_dim_probe, args.seed, tuple(args.hidden))
        print_probe_report(results)
        output.update(results)
        if per_dim_probe is not None:
            os.makedirs(args.fig_dir, exist_ok=True)
            plot_per_dim(results, f"{args.project} #{args.model_num} ({args.epoch}): per-dim probes",
                         f"{args.fig_dir}/per_dim_probes_{tag}.pdf")

    if "factorvae" in args.metrics:
        output["factor_vae"] = run_factor_vae(mu, args.num_repeats, args.seed)
        print_factor_vae_report(output["factor_vae"])

    if "mig" in args.metrics:
        output["mig"] = run_mig(mu, factor_classes, args.num_repeats, args.seed)
        print_mig_report(output["mig"])

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = f"{args.out_dir}/disentanglement_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nsaved results to {out_json}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
