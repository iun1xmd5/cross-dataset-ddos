"""
Model definitions for the transfer experiment.

Each module defines one architecture / estimator and nothing else; the training loops live in
``train.py``. Importing this package does not import TensorFlow or XGBoost - those are loaded
only when the corresponding builder is called.

    from models import NEURAL_BUILDERS, TREE_BUILDERS
    model = NEURAL_BUILDERS["LSTM"]((window, n_features))
    forest = TREE_BUILDERS["Random Forest"](seed=42)
"""

from typing import Callable, Dict

from .bilstm import build_bilstm
from .cnn_lstm import build_cnn_lstm
from .lstm import build_lstm
from .random_forest import build_random_forest
from .xgboost_models import build_xgboost

# display name -> builder taking (window, n_features)
NEURAL_BUILDERS: Dict[str, Callable] = {
    "LSTM": build_lstm,
    "BiLSTM": build_bilstm,
    "CNN-LSTM": build_cnn_lstm,
}

# display name -> builder taking (seed)
TREE_BUILDERS: Dict[str, Callable] = {
    "Random Forest": build_random_forest,
    "XGBoost": build_xgboost,
}

__all__ = [
    "NEURAL_BUILDERS",
    "TREE_BUILDERS",
    "build_lstm",
    "build_bilstm",
    "build_cnn_lstm",
    "build_random_forest",
    "build_xgboost",
]
