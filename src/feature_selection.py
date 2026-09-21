#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared-feature selection between the source and target datasets.

  * compute_shared_features - mutual-information top-k on each dataset, then intersect
  * select_shared_features  - "auto" (MI intersection) or "hardcoded" list from Config
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from config import Config, log


def compute_shared_features(
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    label_col: str,
    top_k: int,
) -> List[str]:
    """Top-``top_k`` features by mutual information in each dataset; return their intersection."""
    numeric_cols = [
        c for c in src_df.columns
        if c not in {label_col, "__binary_label__"}
        and pd.api.types.is_numeric_dtype(src_df[c])
        and c in tgt_df.columns
    ]
    log.info("Intersecting over %d numeric columns", len(numeric_cols))

    def top_mi(df: pd.DataFrame) -> List[str]:
        X = df[numeric_cols].to_numpy(dtype=np.float32)
        y = df["__binary_label__"].to_numpy()
        scores = mutual_info_classif(X, y, random_state=42)
        order = np.argsort(scores)[::-1][:top_k]
        return [numeric_cols[i] for i in order]

    src_top = set(top_mi(src_df))
    tgt_top = set(top_mi(tgt_df))
    shared = sorted(src_top & tgt_top)
    log.info("Shared MI features: %d", len(shared))
    for f in shared:
        log.info("    %s", f)
    return shared


def select_shared_features(src_df: pd.DataFrame, tgt_df: pd.DataFrame, cfg: Config) -> List[str]:
    """Pick the feature set used for one transfer direction, according to ``cfg.feature_mode``."""
    if cfg.feature_mode == "auto":
        shared = compute_shared_features(
            src_df, tgt_df,
            label_col="Label" if "Label" in src_df.columns else "label",
            top_k=cfg.n_features,
        )
        shared = shared[: cfg.shared_feature_count]
    else:
        shared = [
            f for f in cfg.hardcoded_shared_features
            if f in src_df.columns and f in tgt_df.columns
        ]
        log.info("Using hardcoded shared features (%d available)", len(shared))

    if len(shared) < 5:
        raise RuntimeError(f"Too few shared features: {len(shared)}")
    return shared
