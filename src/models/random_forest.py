#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random Forest baseline for flat (one-row-per-flow) features."""

from sklearn.ensemble import RandomForestClassifier

from class_weighting import RF_CLASS_WEIGHT


def build_random_forest(seed: int = 42) -> RandomForestClassifier:
    """
    400 trees, max depth 24, at least 2 samples per leaf. Class imbalance is handled by
    ``class_weight="balanced_subsample"`` (weights recomputed on every bootstrap sample).
    """
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=24,
        min_samples_leaf=2,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=-1,
        random_state=seed,
    )
