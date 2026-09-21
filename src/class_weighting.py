#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class-imbalance handling for every model family.

  * neural networks - per-class weights passed to ``model.fit(class_weight=...)``
  * XGBoost         - per-sample weights ("balanced")
  * Random Forest   - scikit-learn's ``class_weight="balanced_subsample"``

All three implement the same idea: weight each class inversely to its frequency,
so that BENIGN and ATTACK contribute equally to the loss.
"""

from typing import Dict

import numpy as np

RF_CLASS_WEIGHT = "balanced_subsample"


def keras_class_weights(y: np.ndarray) -> Dict[int, float]:
    """{0: total / (2 * n0), 1: total / (2 * n1)} - a class missing from ``y`` gets count 1."""
    y = np.asarray(y)
    n0 = max(int((y == 0).sum()), 1)
    n1 = max(int((y == 1).sum()), 1)
    total = n0 + n1
    return {0: total / (2.0 * n0), 1: total / (2.0 * n1)}


def xgboost_sample_weights(y: np.ndarray) -> np.ndarray:
    """Balanced per-sample weights for ``XGBClassifier.fit(sample_weight=...)``."""
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)
