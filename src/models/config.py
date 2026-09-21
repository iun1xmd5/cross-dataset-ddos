#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central configuration for the zero-shot cross-dataset transfer experiment.

Contains
  * the ``Config`` dataclass (all hyper-parameters in one place),
  * dataset constants (Kaggle slugs, dataset names, model names / file tags),
  * reproducibility + GPU helpers,
  * shared command-line helpers used by train.py, sequence_length_sweep.py and
    computational_cost.py.

TensorFlow is imported lazily (``get_tensorflow``) so that modules which only need
numpy / pandas / scikit-learn (data loading, evaluation, significance tests) stay
fast and work on machines without TensorFlow.
"""

import argparse
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transfer")

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
CICDDOS2019_KAGGLE_SLUG = "rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files"
CICIDS2017_KAGGLE_SLUG = "chethuhn/network-intrusion-dataset"

CICDDOS = "cicddos2019"     # dataset name used in file names / result tags
CICIDS = "cicids2017"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
NEURAL_MODELS = ["LSTM", "BiLSTM", "CNN-LSTM"]
TREE_MODELS = ["Random Forest", "XGBoost"]
ALL_MODELS = NEURAL_MODELS + TREE_MODELS

# display name -> tag used in y_prob_<src>_to_<tgt>_<tag>.npy
MODEL_TAGS: Dict[str, str] = {
    "LSTM": "LSTM",
    "BiLSTM": "BiLSTM",
    "CNN-LSTM": "CNN-LSTM",
    "Random Forest": "RF",
    "XGBoost": "XGB",
}
TAG_TO_MODEL: Dict[str, str] = {v: k for k, v in MODEL_TAGS.items()}


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = 42
    per_class_cap: int = 15_000
    chunk_size: int = 200_000
    n_features: int = 30
    shared_feature_count: int = 22
    window: int = 20
    batch_size: int = 128
    max_epochs: int = 100
    early_stop_patience: int = 12
    lr: float = 1e-3
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 5
    output_dir: Path = Path("results/transfer")
    feature_mode: str = "auto"              # "auto" (mutual information) | "hardcoded"
    use_coral: bool = False                 # CORAL alignment of source -> target features
    eval_balanced_subsample: bool = True    # also evaluate on a balanced target subsample
    models: List[str] = field(default_factory=lambda: list(ALL_MODELS))
    hardcoded_shared_features: List[str] = field(default_factory=lambda: [
        "Average Packet Size",
        "Packet Length Mean",
        "Flow Bytes/s",
        "Flow IAT Mean",
        "Fwd Packets/s",
        "Init_Win_bytes_forward",
        "Flow Duration",
        "Flow Packets/s",
        "Max Packet Length",
        "Min Packet Length",
        "Packet Length Std",
        "Packet Length Variance",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Total",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Fwd Header Length",
        "Subflow Fwd Bytes",
    ])


# ---------------------------------------------------------------------------
# TensorFlow (lazy)
# ---------------------------------------------------------------------------
_TF_STATE: Dict[str, object] = {"checked": False, "module": None, "error": None}


def get_tensorflow():
    """Return the tensorflow module, or None if it cannot be imported."""
    if not _TF_STATE["checked"]:
        _TF_STATE["checked"] = True
        try:
            import tensorflow as tf
            _TF_STATE["module"] = tf
        except Exception as exc:  # pragma: no cover
            _TF_STATE["error"] = str(exc)
    return _TF_STATE["module"]


def tensorflow_import_error() -> Optional[str]:
    get_tensorflow()
    return _TF_STATE["error"]  # type: ignore[return-value]


def require_keras():
    """Return (tf, layers, callbacks, optimizers, regularizers); raise if TensorFlow is missing."""
    tf = get_tensorflow()
    if tf is None:
        raise RuntimeError(f"TensorFlow is not available: {tensorflow_import_error()}")
    from tensorflow.keras import callbacks, layers, optimizers, regularizers
    return tf, layers, callbacks, optimizers, regularizers


# ---------------------------------------------------------------------------
# Reproducibility & GPU
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    # Import TensorFlow BEFORE seeding: importing it (and Keras) consumes Python's global
    # random state, and Keras 3 draws its weight-initialisation seeds from that state.
    # Seeding first and importing later would make runs differ from the original notebook.
    tf = get_tensorflow()
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if tf is not None:
        tf.random.set_seed(seed)
    log.info("Seeds set to %d", seed)


def configure_gpu() -> str:
    """Enable memory growth + mixed precision on GPU. Returns 'GPU', 'CPU' or 'none'."""
    tf = get_tensorflow()
    if tf is None:
        log.warning("TensorFlow unavailable. Neural models will be skipped.")
        return "none"
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        log.warning("No GPU detected. Neural models will run on CPU (slow).")
        return "CPU"
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            log.warning("set_memory_growth failed: %s", exc)
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        log.info("Mixed precision enabled (float16 compute, float32 master weights)")
    except Exception as exc:
        log.warning("Mixed precision setup failed: %s", exc)
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        log.info("Using GPU: %s (%s)", gpu.name, details.get("device_name", "unknown"))
    return "GPU"


# ---------------------------------------------------------------------------
# Shared command-line helpers
# ---------------------------------------------------------------------------
def add_common_args(parser: argparse.ArgumentParser, include_window: bool = True) -> argparse.ArgumentParser:
    """Arguments shared by train.py, sequence_length_sweep.py and computational_cost.py."""
    parser.add_argument("--cicddos-path", type=Path, default=None,
                        help="folder with CICDDoS2019 CSVs (default: download via kagglehub)")
    parser.add_argument("--cicids-path", type=Path, default=None,
                        help="folder with CICIDS2017 CSVs (default: download via kagglehub)")
    parser.add_argument("--output", type=Path, default=Path("results/transfer"),
                        help="results directory (default: results/transfer)")
    parser.add_argument("--feature-mode", choices=["auto", "hardcoded"], default="auto")
    parser.add_argument("--use-coral", action="store_true", default=False,
                        help="align source features to the target with CORAL")
    parser.add_argument("--per-class-cap", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    if include_window:
        parser.add_argument("--window", type=int, default=20,
                            help="sequence length for the neural models (default: 20)")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="maximum training epochs for the neural models (default: 100)")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=None,
                        help="subset of models to run (default: all)")
    return parser


def config_from_args(args: argparse.Namespace, default_models: Optional[List[str]] = None) -> Config:
    models = list(args.models) if args.models else list(default_models or ALL_MODELS)
    return Config(
        seed=args.seed,
        per_class_cap=args.per_class_cap,
        window=getattr(args, "window", 20),
        max_epochs=args.max_epochs,
        output_dir=Path(args.output),
        feature_mode=args.feature_mode,
        use_coral=args.use_coral,
        models=models,
    )
