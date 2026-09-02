import numpy as np
import torch
from sklearn import metrics


def generate_batch_factor_code(encoder, dataloader, num_points, random_state, batch_size):
    """
    Generates representations and factors for a specified number of points.

    Args:
        encoder: PyTorch model (the encoder).
        dataloader: PyTorch DataLoader yielding (images, factors).
        num_points: Number of samples to collect.
        random_state: Numpy random state (unused here but kept for signature compatibility).
        batch_size: Batch size (unused here as dataloader determines it).

    Returns:
        mus: Representations (num_codes, num_points)
        ys: Factors (num_factors, num_points)
    """
    device = next(encoder.parameters()).device
    mus_list = []
    ys_list = []
    points_collected = 0

    encoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            if points_collected >= num_points:
                break

            # Handle different dataloader outputs
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    # Fallback if only data is returned (shouldn't happen based on plan)
                    x = batch[0]
                    y = None
            else:
                x = batch
                y = None

            if y is None:
                raise ValueError("Dataloader must return (image, factor) pairs for disentanglement metrics.")

            x = x.to(device)

            # Encoder expected to return (mu, logvar) or similar. We use mu.
            # Adjust based on specific encoder signature if needed.
            # Based on convnet.py: forward returns mu, logvar
            mu, _ = encoder(x)

            mus_list.append(mu.cpu().numpy())
            ys_list.append(y.cpu().numpy())

            points_collected += x.shape[0]

    if not mus_list:
        raise ValueError("No data collected from dataloader.")

    mus = np.concatenate(mus_list, axis=0)[:num_points]
    ys = np.concatenate(ys_list, axis=0)[:num_points]

    # Transpose to match expected shape (num_codes, num_points) and (num_factors, num_points)
    return mus.T, ys.T


def make_discretizer(target, num_bins=20):
    """
    Creates a discretizer function based on histogram binning.

    Args:
        target: Data to compute bins from (num_codes, num_points).
        num_bins: Number of bins.

    Returns:
        discretizer_fn: Function that takes data and returns discretized indices.
    """
    bins = []
    for i in range(target.shape[0]):
        # Compute bins for each dimension
        # We use np.histogram_bin_edges to get edges
        # But simple linspace min-max might be what disentanglement_lib does.
        # Let's use np.histogram style bins.
        b = np.histogram_bin_edges(target[i], bins=num_bins)
        bins.append(b)

    def discretizer_fn(data):
        # data: (num_codes, num_points)
        discretized = np.zeros_like(data, dtype=int)
        for i in range(data.shape[0]):
            # np.digitize returns indices 1..len(bins)-1. We want 0-indexed.
            # We clip to ensure values outside range fall into first/last bin.
            # digitize: bins[i-1] <= x < bins[i]
            inds = np.digitize(data[i], bins[i]) - 1
            inds = np.clip(inds, 0, num_bins - 1)
            discretized[i] = inds
        return discretized

    return discretizer_fn


def discrete_mutual_info(mus, ys):
    """
    Computes discrete mutual information between representations and factors.

    Args:
        mus: Discretized representations (num_codes, num_points).
        ys: Discrete factors (num_factors, num_points).

    Returns:
        mi_matrix: (num_codes, num_factors)
    """
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    mi_matrix = np.zeros((num_codes, num_factors))

    for i in range(num_codes):
        for j in range(num_factors):
            # Ensure ys[j] is treated as discrete labels by converting to integer codes
            # This handles cases where factors are floats (like dSprites)
            # We use np.unique with return_inverse=True to get integer indices
            _, ys_discrete = np.unique(ys[j], return_inverse=True)
            mi_matrix[i, j] = metrics.mutual_info_score(mus[i], ys_discrete)

    return mi_matrix


def normalize_data(data, mean=None, stddev=None):
    """
    Normalizes data to zero mean and unit variance.

    Args:
        data: (num_dims, num_points)
        mean: Optional precomputed mean.
        stddev: Optional precomputed stddev.

    Returns:
        normalized_data, mean, stddev
    """
    if mean is None:
        mean = np.mean(data, axis=1, keepdims=True)
    if stddev is None:
        stddev = np.std(data, axis=1, keepdims=True)

    # Avoid division by zero
    stddev[stddev == 0] = 1.0

    return (data - mean) / stddev, mean, stddev


def split_train_test(data, train_percentage=None, num_train=None):
    """
    Splits data into training and testing sets.

    Args:
        data: (num_dims, num_points)
        train_percentage: Float between 0 and 1 (optional).
        num_train: Integer (optional).

    Returns:
        train_data, test_data
    """
    num_points = data.shape[1]

    if num_train is None:
        if train_percentage is None:
            raise ValueError("Either train_percentage or num_train must be provided.")
        num_train = int(num_points * train_percentage)

    train_data = data[:, :num_train]
    test_data = data[:, num_train:]

    return train_data, test_data


def obtain_representation(observations, representation_function, batch_size):
    """
    Obtains representation for a set of observations.

    Args:
        observations: (num_points, ...)
        representation_function: Function that takes batch of obs and returns batch of reps.
        batch_size: Batch size.

    Returns:
        representations: (num_codes, num_points)
    """
    # This might not be needed if we use generate_batch_factor_code directly with dataloader
    # But kept for compatibility if needed.
    num_points = observations.shape[0]
    representations = []

    for i in range(0, num_points, batch_size):
        batch = observations[i : i + batch_size]
        rep = representation_function(batch)
        representations.append(rep)

    return np.concatenate(representations, axis=0).T
