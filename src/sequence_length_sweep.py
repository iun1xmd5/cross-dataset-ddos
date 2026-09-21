#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence-length sweep: how does the window length affect zero-shot transfer?

For every transfer direction and every window length, the selected neural model(s) are
re-trained on the source dataset (windows of ``w`` consecutive flows) and evaluated on the
target dataset with the same window length. Tree models do not use windows and are not part
of the sweep.

Feature selection is done once per direction, so the only thing that changes between runs
is the window length.

Usage
    python src/sequence_length_sweep.py \\
        --cicddos-path data/processed/cicddos2019 --cicids-path data/processed/cicids2017 \\
        --windows 5 10 20 30 50 --models LSTM --direction both

Output
    <output>/sequence_length_sweep.csv    one row per direction x model x window
"""

import argparse
import gc
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from config import (
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
    tensorflow_import_error,
)
from data_loader import load_transfer_datasets
from evaluate import evaluate_full, predict_keras
from feature_selection import select_shared_features
from train import NEURAL_BUILDERS, prepare_direction, train_neural

DEFAULT_WINDOWS = [5, 10, 20, 30, 50]
MIN_TRAIN_WINDOWS = 50          # skip window lengths that leave too few training sequences
RESULT_COLUMNS = [
    "direction", "model", "window", "n_train_windows", "n_test_windows", "epochs_run",
    "train_time_s", "roc_auc", "acc_0.5", "acc_opt", "bal_acc_opt", "macro_f1_opt", "thr_opt",
    "score_flipped",
]


def run_sweep(
    cfg: Config,
    windows: List[int],
    models: List[str],
    directions: List[Tuple[str, pd.DataFrame, str, pd.DataFrame]],
) -> pd.DataFrame:
    tf = get_tensorflow()
    if tf is None:
        raise RuntimeError(f"TensorFlow is required for the sweep: {tensorflow_import_error()}")

    rows = []
    for src_name, src_df, tgt_name, tgt_df in directions:
        direction = f"{src_name}->{tgt_name}"
        log.info("=" * 72)
        log.info("Sweep direction: %s", direction)
        log.info("=" * 72)
        shared = select_shared_features(src_df, tgt_df, cfg)      # once per direction

        for window in windows:
            cfg_w = replace(cfg, window=window)
            log.info("-" * 72)
            log.info("Window length = %d", window)
            data = prepare_direction(src_name, src_df, tgt_name, tgt_df, cfg_w, shared=shared)

            if len(data.X_tr_w) < MIN_TRAIN_WINDOWS or len(data.X_tg_w) < 10:
                log.warning("Window %d leaves only %d training / %d test sequences - skipped",
                            window, len(data.X_tr_w), len(data.X_tg_w))
                continue

            for name in models:
                try:
                    model, info = train_neural(
                        name, NEURAL_BUILDERS[name],
                        data.X_tr_w, data.y_tr_w, data.X_va_w, data.y_va_w, cfg_w,
                    )
                    y_prob = predict_keras(model, data.X_tg_w)
                    m = evaluate_full(data.y_tg_w, y_prob, prefix=f"  {name} w={window}")
                    rows.append({
                        "direction": direction,
                        "model": name,
                        "window": window,
                        "n_train_windows": len(data.X_tr_w),
                        "n_test_windows": len(data.X_tg_w),
                        "epochs_run": info["epochs_run"],
                        "train_time_s": round(info["train_time_s"], 2),
                        "roc_auc": m["roc_auc"],
                        "acc_0.5": m["acc_0.5"],
                        "acc_opt": m["acc_opt"],
                        "bal_acc_opt": m["bal_acc_opt"],
                        "macro_f1_opt": m["macro_f1_opt"],
                        "thr_opt": m["thr_opt"],
                        "score_flipped": m["score_flipped"],
                    })
                    del model
                    gc.collect()
                    tf.keras.backend.clear_session()
                except Exception as exc:
                    log.exception("%s (window=%d) failed: %s", name, window, exc)

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No sweep results.")
        return
    for metric in ("roc_auc", "acc_opt"):
        table = df.pivot_table(index=["direction", "model"], columns="window", values=metric)
        print(f"\n{metric} by window length")
        print(table.round(4).to_string())
    best = df.loc[df.groupby(["direction", "model"])["roc_auc"].idxmax()]
    print("\nBest window by ROC-AUC")
    print(best[["direction", "model", "window", "roc_auc", "acc_opt"]].round(4).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser, include_window=False)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS,
                        help=f"window lengths to test (default: {DEFAULT_WINDOWS})")
    parser.add_argument("--direction", choices=["forward", "reverse", "both"], default="both",
                        help="forward = CICDDoS2019->CICIDS2017, reverse = the opposite")
    args = parser.parse_args()

    cfg = config_from_args(args, default_models=["LSTM"])
    models = [m for m in cfg.models if m in NEURAL_MODELS]
    if not models:
        log.error("The sweep only applies to neural models %s", NEURAL_MODELS)
        return 1

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(cfg.seed)
    configure_gpu()

    cicddos, cicids = load_transfer_datasets(args.cicddos_path, args.cicids_path, cfg)
    directions = []
    if args.direction in ("forward", "both"):
        directions.append((CICDDOS, cicddos, CICIDS, cicids))
    if args.direction in ("reverse", "both"):
        directions.append((CICIDS, cicids, CICDDOS, cicddos))

    df = run_sweep(cfg, sorted(set(args.windows)), models, directions)
    out = cfg.output_dir / "sequence_length_sweep.csv"
    df.to_csv(out, index=False)
    log.info("Wrote %s", out)
    print_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
