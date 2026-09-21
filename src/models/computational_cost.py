#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computational cost of every model.

For each transfer direction and model the script measures

  * training time (and epochs run / time per epoch for the neural networks)
  * model size: parameter count (neural) or total tree nodes (Random Forest / XGBoost),
    and the size of the serialised model on disk
  * batch inference throughput (predictions per second and flows per second)
  * single-sample inference latency (median and 95th percentile, in milliseconds)
  * the device the neural networks ran on (CPU / GPU)

A neural prediction consumes one WINDOW of ``window`` flows, while a tree prediction
consumes one flow, so the throughput is reported both per prediction and per flow to make
the two families comparable.

Usage
    python src/computational_cost.py \\
        --cicddos-path data/processed/cicddos2019 --cicids-path data/processed/cicids2017 \\
        --direction forward

Output
    <output>/computational_cost.csv     one row per direction x model

Timing depends heavily on the hardware; compare models measured in the same run.
"""

import argparse
import gc
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

from config import (
    ALL_MODELS,
    CICDDOS,
    CICIDS,
    NEURAL_MODELS,
    Config,
    add_common_args,
    config_from_args,
    configure_gpu,
    get_tensorflow,
    log,
    set_all_seeds,
)
from data_loader import load_transfer_datasets
from evaluate import predict_keras
from models import NEURAL_BUILDERS
from train import (
    DirectionData,
    prepare_direction,
    train_neural,
    train_random_forest,
    train_xgboost,
)

RESULT_COLUMNS = [
    "direction", "model", "device", "train_time_s", "epochs_run", "time_per_epoch_s",
    "n_params", "param_unit", "model_size_mb", "eval_samples", "flows_per_prediction",
    "throughput_pred_per_s", "throughput_flows_per_s",
    "latency_single_ms_median", "latency_single_ms_p95",
]


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def _file_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def serialized_size_mb(name: str, model) -> float:
    """Size of the model serialised to disk, in MB (NaN if it cannot be saved)."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            if name in NEURAL_MODELS:
                path = os.path.join(tmp, "model.keras")
                model.save(path)
            elif name == "XGBoost":
                path = os.path.join(tmp, "model.json")
                model.save_model(path)
            else:
                path = os.path.join(tmp, "model.joblib")
                joblib.dump(model, path)
            return _file_mb(path)
        except Exception as exc:
            log.warning("Could not serialise %s: %s", name, exc)
            return float("nan")


def time_batch_inference(predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, repeats: int = 3) -> float:
    """Median wall time (s) to predict all of X; one warm-up call is excluded."""
    predict(X[: min(len(X), 256)])                                # warm-up (graph build, thread pools)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict(X)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def time_single_inference(predict_one: Callable[[np.ndarray], object], X: np.ndarray, n: int) -> List[float]:
    """Per-call latencies (ms) for ``n`` one-sample predictions."""
    n = min(n, len(X))
    predict_one(X[:1])                                            # warm-up
    lat = []
    for i in range(n):
        x = X[i:i + 1]
        t0 = time.perf_counter()
        predict_one(x)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def measure_direction(
    data: DirectionData,
    cfg: Config,
    models: List[str],
    device: str,
    max_eval_samples: int,
    n_single: int,
) -> List[Dict]:
    tf = get_tensorflow()
    direction = f"{data.source}->{data.target}"
    rows: List[Dict] = []

    for name in models:
        neural = name in NEURAL_MODELS
        if neural and tf is None:
            log.warning("Skipping %s: TensorFlow is not available", name)
            continue
        log.info("-" * 72)
        log.info("Measuring %s (%s)", name, direction)

        try:
            # ---- training -------------------------------------------------
            if neural:
                model, info = train_neural(name, NEURAL_BUILDERS[name],
                                           data.X_tr_w, data.y_tr_w, data.X_va_w, data.y_va_w, cfg)
                X_eval = data.X_tg_w[:max_eval_samples]
                predict = lambda X, m=model: predict_keras(m, X)                      # noqa: E731
                # compiled graph call (as in deployment); .numpy() forces completion on GPU
                infer = tf.function(lambda x, m=model: m(x, training=False), reduce_retracing=True)
                predict_one = lambda x, f=infer: f(tf.constant(x)).numpy()            # noqa: E731
                flows_per_pred = cfg.window
                unit = "parameters"
            else:
                trainer = train_random_forest if name == "Random Forest" else train_xgboost
                model, info = trainer(data.X_tr_f, data.y_tr_f, cfg)
                X_eval = data.X_tg_f[:max_eval_samples]
                predict = lambda X, m=model: m.predict_proba(X)[:, 1]                 # noqa: E731
                predict_one = lambda x, m=model: m.predict_proba(x)                   # noqa: E731
                flows_per_pred = 1
                unit = "tree nodes"

            # ---- inference ------------------------------------------------
            batch_s = time_batch_inference(predict, X_eval)
            pred_per_s = len(X_eval) / batch_s
            lat = time_single_inference(predict_one, X_eval, n_single)

            epochs = info.get("epochs_run")
            rows.append({
                "direction": direction,
                "model": name,
                "device": device if neural else "CPU",
                "train_time_s": round(info["train_time_s"], 2),
                "epochs_run": epochs,
                "time_per_epoch_s": round(info["train_time_s"] / epochs, 3) if epochs else None,
                "n_params": info["n_params"],
                "param_unit": unit,
                "model_size_mb": round(serialized_size_mb(name, model), 3),
                "eval_samples": len(X_eval),
                "flows_per_prediction": flows_per_pred,
                "throughput_pred_per_s": round(pred_per_s, 1),
                "throughput_flows_per_s": round(pred_per_s * flows_per_pred, 1),
                "latency_single_ms_median": round(statistics.median(lat), 3),
                "latency_single_ms_p95": round(float(np.percentile(lat, 95)), 3),
            })
            log.info("  %s: train %.1fs | %d %s | %.0f pred/s | %.2f ms/sample",
                     name, info["train_time_s"], info["n_params"], unit, pred_per_s,
                     statistics.median(lat))

            del model
            gc.collect()
            if neural:
                tf.keras.backend.clear_session()
        except ImportError as exc:
            log.warning("Skipping %s: %s", name, exc)
        except Exception as exc:
            log.exception("%s failed: %s", name, exc)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument("--direction", choices=["forward", "reverse", "both"], default="forward",
                        help="forward = CICDDoS2019->CICIDS2017 (default), reverse = the opposite")
    parser.add_argument("--max-eval-samples", type=int, default=20_000,
                        help="cap on the number of target samples used for the throughput test")
    parser.add_argument("--n-single", type=int, default=100,
                        help="number of single-sample calls used for the latency statistics")
    args = parser.parse_args()

    cfg = config_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(cfg.seed)
    device = configure_gpu()

    cicddos, cicids = load_transfer_datasets(args.cicddos_path, args.cicids_path, cfg)
    directions = []
    if args.direction in ("forward", "both"):
        directions.append((CICDDOS, cicddos, CICIDS, cicids))
    if args.direction in ("reverse", "both"):
        directions.append((CICIDS, cicids, CICDDOS, cicddos))

    selected = [m for m in ALL_MODELS if m in cfg.models]
    rows: List[Dict] = []
    for src_name, src_df, tgt_name, tgt_df in directions:
        log.info("=" * 72)
        log.info("Cost measurement: %s -> %s", src_name, tgt_name)
        log.info("=" * 72)
        data = prepare_direction(src_name, src_df, tgt_name, tgt_df, cfg)
        rows += measure_direction(data, cfg, selected, device, args.max_eval_samples, args.n_single)

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    out = cfg.output_dir / "computational_cost.csv"
    df.to_csv(out, index=False)
    log.info("Wrote %s", out)
    print("\n" + df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
