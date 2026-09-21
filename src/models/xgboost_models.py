#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost baseline for flat (one-row-per-flow) features.

The file is called ``xgboost_models.py`` (not ``xgboost.py``) so it never shadows the
``xgboost`` package it imports.
"""


def build_xgboost(seed: int = 42):
    """
    400 histogram-based trees of depth 8, learning rate 0.08, 90% row / column subsampling.
    Class imbalance is handled with per-sample weights at fit time (see class_weighting.py).
    Raises ImportError if the ``xgboost`` package is not installed.
    """
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=seed,
        eval_metric="logloss",
        reg_lambda=1.0,
    )
