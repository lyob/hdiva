# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.stats
from sklearn import ensemble, preprocessing

from d_analysis.metrics import utils


def compute_dci(
    encoder,
    dataloader,
    num_train,
    num_test,
    random_state=None,
    batch_size=None,
):
    """Computes the DCI scores according to Sec 2.

    Args:
      encoder: PyTorch model (the encoder).
      dataloader: PyTorch DataLoader yielding (images, factors).
      num_train: Number of points used for training.
      num_test: Number of points used for testing.
      random_state: Numpy random state used for randomness.
      batch_size: Batch size for sampling (unused, determined by dataloader).

    Returns:
      Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """
    # mus_train are of shape [num_codes, num_train], while ys_train are of shape
    # [num_factors, num_train].
    mus_train, ys_train = utils.generate_batch_factor_code(encoder, dataloader, num_train, random_state, batch_size)
    assert mus_train.shape[1] == num_train
    assert ys_train.shape[1] == num_train

    # For test set, we need to continue iterating from dataloader or restart it
    # Ideally dataloader is infinite or large enough.
    # If dataloader is exhausted, we might need to re-initialize it or handle it.
    # Assuming dataloader is sufficient for now.
    mus_test, ys_test = utils.generate_batch_factor_code(encoder, dataloader, num_test, random_state, batch_size)

    scores = _compute_dci(mus_train, ys_train, mus_test, ys_test)
    return scores


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(mus_train, ys_train, mus_test, ys_test)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        if len(np.unique(y_train[i, :])) < 2:
            # Factor is constant, cannot train classifier.
            # Assign 0 importance and 1.0 accuracy (predicting the constant value).
            importance_matrix[:, i] = 0.0
            train_loss.append(1.0)
            test_loss.append(1.0)
            continue

        model = ensemble.GradientBoostingClassifier()
        # Discretize target using LabelEncoder
        le = preprocessing.LabelEncoder()
        y_train_enc = le.fit_transform(y_train[i, :])
        y_test_enc = le.transform(y_test[i, :])

        model.fit(x_train.T, y_train_enc)
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train_enc))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test_enc))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1.0 - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1.0 - scipy.stats.entropy(importance_matrix + 1e-11, base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """ "Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)
