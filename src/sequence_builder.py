#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence construction for the neural models (LSTM, BiLSTM, CNN-LSTM).

Flow records are grouped into non-overlapping windows of ``window`` consecutive rows.
Each window becomes one sample of shape (window, n_features) and is labelled with the
label of its LAST flow. Trailing rows that do not fill a window are discarded.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_windows(
    df: pd.DataFrame,
    features: List[str],
    scaler: MinMaxScaler,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_win, y_win) with X_win.shape == (n_windows, window, n_features)."""
    X = scaler.transform(df[features].to_numpy(dtype=np.float32))
    y = df["__binary_label__"].to_numpy(dtype=np.int32)
    n_win = len(X) // window
    X_win = np.empty((n_win, window, X.shape[1]), dtype=np.float32)
    y_win = np.empty(n_win, dtype=np.int32)
    for i in range(n_win):
        s, e = i * window, (i + 1) * window
        X_win[i] = X[s:e]
        y_win[i] = y[e - 1]          # label of last timestep in window
    return X_win, y_win
