#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing utilities.

  * split_stratified          - stratified train / validation / test split (70/15/15)
  * fit_scaler                - MinMaxScaler fitted on the SOURCE training split only
  * build_flat                - scaled feature matrix + labels (Random Forest / XGBoost input)
  * coral_align               - CORAL: match source second-order statistics to the target
  * balanced_indices          - indices of a class-balanced subsample
  * make_balanced_subsample   - convenience wrapper returning the subsampled arrays

Sequence windows for the neural models live in sequence_builder.py.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def split_stratified(df: pd.DataFrame, ratios=(0.7, 0.15, 0.15)):
    """Stratified split on the binary label. Returns (train, val, test) DataFrames."""
    from sklearn.model_selection import train_test_split

    train_ratio, val_ratio, test_ratio = ratios
    train, temp = train_test_split(
        df, test_size=1 - train_ratio,
        stratify=df["__binary_label__"], random_state=42,
    )
    val, test = train_test_split(
        temp, test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp["__binary_label__"], random_state=42,
    )
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def fit_scaler(df: pd.DataFrame, features: List[str]) -> MinMaxScaler:
    """Fit a MinMaxScaler on the given (source-train) rows."""
    scaler = MinMaxScaler()
    scaler.fit(df[features].to_numpy(dtype=np.float32))
    return scaler


def build_flat(df: pd.DataFrame, features: List[str], scaler: MinMaxScaler):
    """Scaled 2-D feature matrix and binary labels (one row per flow)."""
    X = scaler.transform(df[features].to_numpy(dtype=np.float32))
    y = df["__binary_label__"].to_numpy(dtype=np.int32)
    return X, y


def coral_align(X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
    """
    Simple CORAL: align second-order statistics of source to target.
    Returns the transformed source features; the target is left unchanged.
    """
    def _cov(X):
        X = X - X.mean(axis=0, keepdims=True)
        return (X.T @ X) / max(X.shape[0] - 1, 1)

    cs = _cov(X_src) + np.eye(X_src.shape[1]) * 1e-6
    ct = _cov(X_tgt) + np.eye(X_tgt.shape[1]) * 1e-6

    # Whitening + recolouring via SVD for numerical stability
    us, ss, _ = np.linalg.svd(cs)
    ut, st, _ = np.linalg.svd(ct)
    ss_inv_sqrt = np.diag(1.0 / np.sqrt(ss + 1e-8))
    st_sqrt = np.diag(np.sqrt(st + 1e-8))
    A = us @ ss_inv_sqrt @ us.T @ ut @ st_sqrt @ ut.T
    X_src_aligned = (X_src - X_src.mean(axis=0)) @ A + X_tgt.mean(axis=0)
    return X_src_aligned.astype(np.float32)


def balanced_indices(y: np.ndarray, seed: int = 42) -> Optional[np.ndarray]:
    """
    Indices of a balanced subsample (equal number of class 0 and class 1).
    The selection depends only on ``y`` and ``seed``, so it can be reproduced later from
    the saved ``y_true_*.npy`` files. Returns None if a class is missing.
    """
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    if n == 0:
        return None
    sel0 = rng.choice(idx0, n, replace=False)
    sel1 = rng.choice(idx1, n, replace=False)
    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)
    return sel


def make_balanced_subsample(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Return a balanced subsample (X, y); returns the inputs unchanged if a class is missing."""
    sel = balanced_indices(y, seed)
    if sel is None:
        return X, y
    return X[sel], y[sel]
