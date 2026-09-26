"""FactorVAE disentanglement score (Kim & Mnih, 2018, https://arxiv.org/abs/1802.05983).

Follows the disentanglement_lib implementation (batch size 64, 10000 train / 5000 eval votes,
10000 samples for the global variances, dims with variance < 0.05 pruned). Instead of re-encoding
images for every vote, it takes precomputed codes for every image of a dataset laid out as a full
factor grid (like dSprites), so a batch with one factor held fixed is an index lookup.

Each vote: pick a factor k, sample a batch of factor vectors that share factor k, and record the
latent dim with the smallest variance across the batch (after dividing by its global variance).
A majority-vote classifier maps each latent dim to a factor; the score is its accuracy.
"""

import numpy as np


def compute_factor_vae(
    codes,
    factor_sizes,
    factor_strides,
    random_state,
    batch_size=64,
    num_train=10000,
    num_eval=5000,
    num_variance_estimate=10000,
    prune_threshold=0.05,
):
    """
    Args:
        codes: (num_images, num_codes) representation (e.g. posterior means) of every image,
            where image index = sum(factor_class * factor_stride).
        factor_sizes: number of values of each factor, e.g. [3, 6, 40, 32, 32] for dSprites.
        factor_strides: index stride of each factor in `codes`.
        random_state: np.random.RandomState.

    Returns:
        dict with train_accuracy, eval_accuracy, num_active_dims, active_dims, and
        dim_to_factor (majority-vote factor index for each active dim).
    """
    factor_sizes = np.asarray(factor_sizes)
    factor_strides = np.asarray(factor_strides)

    def sample_indices(n):
        factors = random_state.randint(0, factor_sizes, size=(n, len(factor_sizes)))
        return factors @ factor_strides

    global_variances = np.var(codes[sample_indices(num_variance_estimate)], axis=0, ddof=1)
    active_dims = global_variances >= prune_threshold
    if not active_dims.any():
        return {"train_accuracy": 0.0, "eval_accuracy": 0.0, "num_active_dims": 0,
                "active_dims": [], "dim_to_factor": []}

    def generate_votes(num_votes):
        factor_index = random_state.randint(len(factor_sizes), size=num_votes)
        factors = random_state.randint(0, factor_sizes, size=(num_votes, batch_size, len(factor_sizes)))
        rows = np.arange(num_votes)
        factors[rows, :, factor_index] = factors[rows, 0, factor_index][:, None]  # fix factor k across the batch
        local_variances = np.var(codes[factors @ factor_strides], axis=1, ddof=1)  # (num_votes, num_codes)
        argmin = np.argmin(local_variances[:, active_dims] / global_variances[active_dims], axis=1)
        votes = np.zeros((len(factor_sizes), active_dims.sum()), dtype=np.int64)
        np.add.at(votes, (factor_index, argmin), 1)
        return votes

    training_votes = generate_votes(num_train)
    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])
    train_accuracy = training_votes[classifier, other_index].sum() / training_votes.sum()

    eval_votes = generate_votes(num_eval)
    eval_accuracy = eval_votes[classifier, other_index].sum() / eval_votes.sum()

    return {
        "train_accuracy": float(train_accuracy),
        "eval_accuracy": float(eval_accuracy),
        "num_active_dims": int(active_dims.sum()),
        "active_dims": np.flatnonzero(active_dims).tolist(),
        "dim_to_factor": classifier.tolist(),
    }
