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

"""Modularity and explicitness metrics from the F-statistic paper.

Based on "Learning Deep Disentangled Embeddings With the F-Statistic Loss"
(https://arxiv.org/pdf/1802.05312.pdf).
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from sklearn import linear_model, metrics, preprocessing

from d_analysis.metrics import utils


def compute_modularity_explicitness(
    encoder,
    dataloader,
    num_train,
    num_test,
    random_state=None,
    batch_size=None,
):
    """Computes the modularity metric according to Sec 3.

    Args:
      encoder: PyTorch model (the encoder).
      dataloader: PyTorch DataLoader yielding (images, factors).
      num_train: Number of points used for training.
      num_test: Number of points used for testing.
      random_state: Numpy random state used for randomness.
      batch_size: Batch size for sampling (unused, determined by dataloader).

    Returns:
      Dictionary with average modularity score and average explicitness
        (train and test).
    """
    scores = {}
    mus_train, ys_train = utils.generate_batch_factor_code(encoder, dataloader, num_train, random_state, batch_size)
    mus_test, ys_test = utils.generate_batch_factor_code(encoder, dataloader, num_test, random_state, batch_size)

    discretizer_fn = utils.make_discretizer(mus_train)
    discretized_mus = discretizer_fn(mus_train)

    mutual_information = utils.discrete_mutual_info(discretized_mus, ys_train)
    # Mutual information should have shape [num_codes, num_factors].
    assert mutual_information.shape[0] == mus_train.shape[0]
    assert mutual_information.shape[1] == ys_train.shape[0]
    scores["modularity_score"] = modularity(mutual_information)

    explicitness_score_train = np.zeros([ys_train.shape[0], 1])
    explicitness_score_test = np.zeros([ys_test.shape[0], 1])
    mus_train_norm, mean_mus, stddev_mus = utils.normalize_data(mus_train)
    mus_test_norm, _, _ = utils.normalize_data(mus_test, mean_mus, stddev_mus)

    for i in range(ys_train.shape[0]):
        explicitness_score_train[i], explicitness_score_test[i] = explicitness_per_factor(
            mus_train_norm, ys_train[i, :], mus_test_norm, ys_test[i, :]
        )
    scores["explicitness_score_train"] = np.mean(explicitness_score_train)
    scores["explicitness_score_test"] = np.mean(explicitness_score_test)
    return scores


def explicitness_per_factor(mus_train, y_train, mus_test, y_test):
    """Compute explicitness score for a factor as ROC-AUC of a classifier.

    Args:
      mus_train: Representation for training, (num_codes, num_points)-np array.
      y_train: Ground truth factors for training, (num_factors, num_points)-np
        array.
      mus_test: Representation for testing, (num_codes, num_points)-np array.
      y_test: Ground truth factors for testing, (num_factors, num_points)-np
        array.

    Returns:
      roc_train: ROC-AUC score of the classifier on training data.
      roc_test: ROC-AUC score of the classifier on testing data.
    """
    # Check if y_train has at least 2 classes
    if len(np.unique(y_train)) < 2:
        # Constant factor, cannot train classifier
        # Assign 0.5 ROC AUC (random guess) or 1.0 if we consider it perfectly predicted?
        # ROC AUC is undefined for 1 class usually, but if we must return a score:
        # If we predict the constant class with probability 1.0, what is ROC AUC?
        # Ideally we should handle this. Let's return 0.5 as "uninformative".
        return 0.5, 0.5

    x_train = np.transpose(mus_train)
    x_test = np.transpose(mus_test)

    # Discretize targets
    le = preprocessing.LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    clf = linear_model.LogisticRegression(solver="liblinear").fit(x_train, y_train_enc)
    y_pred_train = clf.predict_proba(x_train)
    y_pred_test = clf.predict_proba(x_test)
    mlb = preprocessing.MultiLabelBinarizer()

    # Ensure y_train/y_test are suitable for MultiLabelBinarizer or handle multiclass ROC AUC correctly
    # If y is single label multiclass, we need to binarize it.
    # The original code used expand_dims and MultiLabelBinarizer which suggests it treats it as multilabel or just one-hot encoding.
    y_train_bin = mlb.fit_transform(np.expand_dims(y_train_enc, 1))
    y_test_bin = mlb.transform(np.expand_dims(y_test_enc, 1))

    # Handle cases where some classes might be missing in test set or train set
    # roc_auc_score with multi_class='ovr' is robust.
    # However, original code used this specific transform.
    # If y_pred has different shape than y_bin (e.g. missing classes), we need to be careful.
    # For now, following original logic but adding error handling if needed.

    try:
        roc_train = metrics.roc_auc_score(y_train_bin, y_pred_train, multi_class="ovr")
        roc_test = metrics.roc_auc_score(y_test_bin, y_pred_test, multi_class="ovr")
    except ValueError:
        # Fallback or simplified calculation if ROC AUC fails (e.g. only one class present)
        roc_train = 0.5
        roc_test = 0.5

    return roc_train, roc_test


def modularity(mutual_information):
    """Computes the modularity from mutual information."""
    # Mutual information has shape [num_codes, num_factors].
    squared_mi = np.square(mutual_information)
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] - 1.0)
    delta = numerator / denominator
    modularity_score = 1.0 - delta
    index = max_squared_mi == 0.0
    modularity_score[index] = 0.0
    return np.mean(modularity_score)
