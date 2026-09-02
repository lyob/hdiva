import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def sample_gaussian_scale_mixture(num_samples: int = 1000, seed: int = 42) -> Tensor:
    """
    Sample from a Gaussian scale mixture distribution with 3 hidden factors.

    The generative process is as follows:
        z ~ LogNormal(0, sigma_z^2)
        y|z ~ N(0, z * diag(lambda_1, lambda_2))
        x|y ~ N(y, sigma_x^2 * I)

    Args:
        num_samples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        Tensor: Samples from the distribution with shape (num_samples, 2).
    """
    np.random.seed(seed)

    sigma_z = 0.5
    lambda_1 = 0.1
    lambda_2 = 1.0
    sigma_x = 0.05

    # z ~ LogNormal(0, sigma_z^2)
    z = np.random.lognormal(0, sigma_z, num_samples)

    # y|z ~ N(0, z * diag(lambda_1, lambda_2))
    y = np.zeros((num_samples, 2))
    for i in range(num_samples):
        cov_matrix = z[i] * np.array([[lambda_1, 0], [0, lambda_2]])
        y[i] = np.random.multivariate_normal([0, 0], cov_matrix)

    # x|y ~ N(y, sigma_x^2 * I)
    x = np.zeros((num_samples, 2))
    for i in range(num_samples):
        cov_matrix = np.eye(2) * sigma_x**2
        x[i] = np.random.multivariate_normal(y[i], cov_matrix)

    return torch.from_numpy(x).float()