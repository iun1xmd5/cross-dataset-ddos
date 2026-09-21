#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot cross-dataset transfer experiment for DDoS detection.

Train on one dataset, test - without any fine-tuning - on the other, in both directions
(CICDDoS2019 -> CICIDS2017 and CICIDS2017 -> CICDDoS2019), with

    LSTM, BiLSTM, CNN-LSTM      (sequence windows)
    Random Forest, XGBoost      (flat flow features)

Pipeline per direction
    shared features -> stratified split of the source -> scaler fitted on source-train only
    -> windows / flat matrices -> (optional CORAL) -> train on source -> evaluate on the target

This file holds the training loops and the experiment driver. Everything else lives in
the sibling modules: config.py, data_loader.py, feature_selection.py, preprocessing.py,
sequence_builder.py, class_weighting.py, evaluate.py and the model definitions in models/.

Usage
    python src/train.py --cicddos-path data/processed/cicddos2019 \\
                        --cicids-path  data/processed/cicids2017 \\
                        --output results/transfer [--use-coral] [--models "Random Forest" XGBoost]

From a notebook
    import sys; sys.path.append("src")
    from train import run_experiment
    results = run_experiment(output="results/transfer", use_coral=False)
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from class_weighting import keras_class_weights, xgboost_sample_weights
from config import (
    ALL_MODELS,
    NEURAL_MODELS,
    Config,
    add_common_args,
    config_from_args,
    configure_gpu,
    get_tensorflow,
    log,
    require_keras,
    set_all_seeds,
    tensorflow_import_error,
    CICDDOS,
    CICIDS,
)
from data_loader import load_transfer_datasets
from evaluate import (
    DirectionResult,
    predict_keras,
    print_paper_table,
    record_result,
    write_results_csv,
)
from feature_selection import select_shared_features
from models import NEURAL_BUILDERS, build_random_forest, build_xgboost
from preprocessing import build_flat, coral_align, fit_scaler, split_stratified
from sequence_builder import build_windows


def compile_model(model, lr: float):
    _, _, _, optimizers, _ = require_keras()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ===========================================================================
# Training  (every trainer returns (model, info) with info["train_time_s"])
# ===========================================================================
def train_neural(model_name: str, builder: Callable, X_tr, y_tr, X_va, y_va, cfg: Config):
    _, _, callbacks, _, _ = require_keras()
    log.info("Training %s on source data (shape=%s)", model_name, X_tr.shape)
    model = builder((X_tr.shape[1], X_tr.shape[2]))
    compile_model(model, cfg.lr)

    class_weight = keras_class_weights(y_tr)
    log.info("  class_weight = %s", class_weight)

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.early_stop_patience,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=cfg.lr_reduce_factor,
            patience=cfg.lr_reduce_patience, min_lr=1e-6,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        callbacks=cb,
        class_weight=class_weight,
        verbose=2,
    )
    elapsed = time.time() - t0
    log.info("%s trained in %.1f s", model_name, elapsed)
    info = {
        "train_time_s": elapsed,
        "epochs_run": len(history.history["loss"]),
        "n_params": int(model.count_params()),
    }
    return model, info


def train_random_forest(X_tr, y_tr, cfg: Config):
    log.info("Training Random Forest on source data (shape=%s)", X_tr.shape)
    t0 = time.time()
    rf = build_random_forest(cfg.seed)
    rf.fit(X_tr, y_tr)
    info = {
        "train_time_s": time.time() - t0,
        "n_params": int(sum(e.tree_.node_count for e in rf.estimators_)),   # total tree nodes
    }
    return rf, info


def train_xgboost(X_tr, y_tr, cfg: Config):
    log.info("Training XGBoost on source data (shape=%s)", X_tr.shape)
    t0 = time.time()
    clf = build_xgboost(cfg.seed)
    clf.fit(X_tr, y_tr, sample_weight=xgboost_sample_weights(y_tr))
    info = {
        "train_time_s": time.time() - t0,
        "n_params": int(len(clf.get_booster().trees_to_dataframe())),        # total tree nodes
    }
    return clf, info


# ===========================================================================
# Per-direction data preparation (shared with sequence_length_sweep.py and
# computational_cost.py)
# ===========================================================================
@dataclass
class DirectionData:
    source: str
    target: str
    shared: List[str]
    scaler: MinMaxScaler
    # source, windowed (neural) and flat (trees)
    X_tr_w: np.ndarray
    y_tr_w: np.ndarray
    X_va_w: np.ndarray
    y_va_w: np.ndarray
    X_tr_f: np.ndarray
    y_tr_f: np.ndarray
    X_va_f: np.ndarray
    y_va_f: np.ndarray
    # target (test), windowed and flat
    X_tg_w: np.ndarray
    y_tg_w: np.ndarray
    X_tg_f: np.ndarray
    y_tg_f: np.ndarray


def prepare_direction(
    src_name: str,
    src_df: pd.DataFrame,
    tgt_name: str,
    tgt_df: pd.DataFrame,
    cfg: Config,
    shared: Optional[List[str]] = None,
) -> DirectionData:
    """Feature selection, split, scaling, windowing and optional CORAL for one direction."""
    # 1. Feature selection (pass ``shared`` to reuse a previous selection)
    if shared is None:
        shared = select_shared_features(src_df, tgt_df, cfg)

    # 2. Split source
    src_train, src_val, _ = split_stratified(src_df)
    log.info("Source split: train=%d val=%d", len(src_train), len(src_val))

    # 3. Scaler on source train only
    scaler = fit_scaler(src_train, shared)

    # 4. Tensors
    X_tr_w, y_tr_w = build_windows(src_train, shared, scaler, cfg.window)
    X_va_w, y_va_w = build_windows(src_val, shared, scaler, cfg.window)
    X_tr_f, y_tr_f = build_flat(src_train, shared, scaler)
    X_va_f, y_va_f = build_flat(src_val, shared, scaler)

    X_tg_w, y_tg_w = build_windows(tgt_df, shared, scaler, cfg.window)
    X_tg_f, y_tg_f = build_flat(tgt_df, shared, scaler)
    log.info("Target test shape: windowed=%s flat=%s", X_tg_w.shape, X_tg_f.shape)
    log.info(
        "Target class balance: BENIGN=%d  ATTACK=%d  (%.1f%% attack)",
        int((y_tg_f == 0).sum()), int((y_tg_f == 1).sum()),
        100.0 * (y_tg_f == 1).mean(),
    )

    # Optional CORAL alignment (flat features only; windows are left as they are)
    if cfg.use_coral:
        log.info("Applying CORAL alignment on flat features...")
        X_tr_f = coral_align(X_tr_f, X_tg_f)
        X_va_f = coral_align(X_va_f, X_tg_f)

    return DirectionData(
        source=src_name, target=tgt_name, shared=shared, scaler=scaler,
        X_tr_w=X_tr_w, y_tr_w=y_tr_w, X_va_w=X_va_w, y_va_w=y_va_w,
        X_tr_f=X_tr_f, y_tr_f=y_tr_f, X_va_f=X_va_f, y_va_f=y_va_f,
        X_tg_w=X_tg_w, y_tg_w=y_tg_w, X_tg_f=X_tg_f, y_tg_f=y_tg_f,
    )


# ===========================================================================
# Direction runner
# ===========================================================================
def run_direction(
    src_name: str,
    src_df: pd.DataFrame,
    tgt_name: str,
    tgt_df: pd.DataFrame,
    cfg: Config,
) -> DirectionResult:
    log.info("=" * 72)
    log.info("Direction: %s -> %s", src_name, tgt_name)
    log.info("=" * 72)

    data = prepare_direction(src_name, src_df, tgt_name, tgt_df, cfg)
    result = DirectionResult(source=src_name, target=tgt_name, shared_features=data.shared)
    selected = [m for m in ALL_MODELS if m in cfg.models]

    def record(name: str, y_true, y_prob):
        record_result(result, name, y_true, y_prob, cfg.output_dir,
                      seed=cfg.seed, eval_balanced=cfg.eval_balanced_subsample)

    # Neural models
    tf = get_tensorflow()
    for name in [m for m in selected if m in NEURAL_MODELS]:
        if tf is None:
            log.warning("Skipping %s: %s", name, tensorflow_import_error())
            continue
        try:
            model, _ = train_neural(name, NEURAL_BUILDERS[name],
                                    data.X_tr_w, data.y_tr_w, data.X_va_w, data.y_va_w, cfg)
            record(name, data.y_tg_w, predict_keras(model, data.X_tg_w))
            del model
            gc.collect()
            tf.keras.backend.clear_session()
        except Exception as exc:
            log.exception("%s training failed: %s", name, exc)

    # Random Forest
    if "Random Forest" in selected:
        try:
            rf, _ = train_random_forest(data.X_tr_f, data.y_tr_f, cfg)
            record("Random Forest", data.y_tg_f, rf.predict_proba(data.X_tg_f)[:, 1])
            del rf
            gc.collect()
        except Exception as exc:
            log.exception("RF training failed: %s", exc)

    # XGBoost
    if "XGBoost" in selected:
        try:
            xg, _ = train_xgboost(data.X_tr_f, data.y_tr_f, cfg)
            record("XGBoost", data.y_tg_f, xg.predict_proba(data.X_tg_f)[:, 1])
            del xg
            gc.collect()
        except ImportError:
            log.warning("Skipping XGBoost: package not installed")
        except Exception as exc:
            log.exception("XGBoost training failed: %s", exc)

    # Persist artefacts
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{src_name}_to_{tgt_name}"
    joblib.dump(data.scaler, cfg.output_dir / f"scaler_{tag}.joblib")
    with open(cfg.output_dir / f"shared_features_{tag}.json", "w") as fh:
        json.dump(data.shared, fh, indent=2)

    return result


# ===========================================================================
# Main entry
# ===========================================================================
def run_experiment_from_config(
    cfg: Config,
    cicddos_path: Optional[Path] = None,
    cicids_path: Optional[Path] = None,
) -> List[DirectionResult]:
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(cfg.seed)
    configure_gpu()

    cicddos, cicids = load_transfer_datasets(cicddos_path, cicids_path, cfg)

    results: List[DirectionResult] = [
        run_direction(CICDDOS, cicddos, CICIDS, cicids, cfg),
        run_direction(CICIDS, cicids, CICDDOS, cicddos, cfg),
    ]

    write_results_csv(results, cfg.output_dir)
    print_paper_table(results)
    return results


def run_experiment(
    cicddos_path: Optional[Path] = None,
    cicids_path: Optional[Path] = None,
    output: Path = Path("results/transfer"),
    feature_mode: str = "auto",
    use_coral: bool = False,
    per_class_cap: int = 15_000,
    seed: int = 42,
    **overrides,
) -> List[DirectionResult]:
    """
    Notebook-friendly wrapper. Extra keyword arguments override any ``Config`` field,
    e.g. ``max_epochs=10``, ``window=30`` or ``models=["Random Forest", "XGBoost"]``.
    """
    cfg = Config(
        seed=seed, per_class_cap=per_class_cap, feature_mode=feature_mode,
        use_coral=use_coral, output_dir=Path(output),
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"Unknown Config field: {key}")
        setattr(cfg, key, value)
    return run_experiment_from_config(
        cfg,
        Path(cicddos_path) if cicddos_path is not None else None,
        Path(cicids_path) if cicids_path is not None else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    args, unknown = parser.parse_known_args()
    if unknown:
        log.warning("Ignoring unrecognized arguments: %s", unknown)

    cfg = config_from_args(args)
    run_experiment_from_config(cfg, args.cicddos_path, args.cicids_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
