import numpy as np

from d_analysis.metrics import dci, modularity


def measure_disentanglement(encoder, dataloader, metric_name, num_train, num_test, random_seed: int | None = None):
    """
    Measures disentanglement using the specified metric.

    Args:
        encoder: PyTorch model (the encoder).
        dataloader: PyTorch DataLoader yielding (images, factors).
        metric_name: String, either "DCI" or "Modularity".
        num_train: Number of training samples.
        num_test: Number of testing samples.
        random_seed: Integer seed for random state used for randomness.

    Returns:
        Dictionary containing metric scores.
    """
    if random_seed is None:
        random_seed = np.random.RandomState(42)
    else:
        random_seed = np.random.RandomState(random_seed)

    if metric_name == "DCI":
        scores = dci.compute_dci(
            encoder=encoder, dataloader=dataloader, num_train=num_train, num_test=num_test, random_state=random_seed
        )
    elif metric_name == "Modularity":
        scores = modularity.compute_modularity_explicitness(
            encoder=encoder, dataloader=dataloader, num_train=num_train, num_test=num_test, random_state=random_seed
        )
    else:
        raise ValueError(f"Unknown metric name: {metric_name}. Supported metrics: 'DCI', 'Modularity'.")

    return scores
