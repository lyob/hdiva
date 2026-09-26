"""Mutual Information Gap (Chen et al., 2018, https://arxiv.org/abs/1802.04942).

Follows the disentanglement_lib implementation: each code dim is discretized into 20 histogram
bins, the discrete mutual information I(z_j; v_k) is computed for every (code, factor) pair, and
MIG = mean over factors of (largest MI - second-largest MI) / H(v_k).
"""

import numpy as np
from sklearn.metrics import mutual_info_score

from d_analysis.metrics import utils


def compute_mig(codes, factor_classes, num_bins=20):
    """
    Args:
        codes: (num_points, num_codes) representation, e.g. posterior means.
        factor_classes: (num_points, num_factors) integer factor values.
        num_bins: histogram bins per code dim.

    Returns:
        dict with mig, mig_per_factor, mi_matrix (num_codes, num_factors), factor_entropy.
    """
    mus = codes.T
    ys = factor_classes.T
    discretized = utils.make_discretizer(mus, num_bins)(mus)
    mi = utils.discrete_mutual_info(discretized, ys)
    entropy = np.array([mutual_info_score(y, y) for y in ys])
    sorted_mi = np.sort(mi, axis=0)[::-1]
    per_factor = (sorted_mi[0] - sorted_mi[1]) / entropy
    return {
        "mig": float(per_factor.mean()),
        "mig_per_factor": per_factor.tolist(),
        "mi_matrix": mi.tolist(),
        "factor_entropy": entropy.tolist(),
    }
