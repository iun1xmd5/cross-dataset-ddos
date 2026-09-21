#!/usr/bin/env bash
# =============================================================================
# Statistics only: recompute every metric from the predictions saved by a previous
# run (results/transfer/y_prob_*.npy + y_true_*.npy). No datasets are read and no
# model is trained, so this takes seconds.
#
# Produces, in the results directory:
#   transfer_results_recomputed.csv   AUC, Acc/BalAcc/macro-F1 @0.5 and @optimal threshold,
#                                     score-inversion diagnostic, balanced-subsample metrics
#   transfer_statistics.json          the same plus confusion matrices and probability summaries
# and prints the paper-style table to the terminal.
#
# Usage:
#   bash scripts/run_statistics_only.sh
#   OUTPUT_DIR=/kaggle/working/results/transfer bash scripts/run_statistics_only.sh
#
# Environment variables (all optional):
#   PYTHON       python interpreter                      (default: python3 / python)
#   OUTPUT_DIR   directory holding y_prob_*/y_true_*.npy  (default: <repo>/results/transfer)
#   SEED         seed used for the balanced subsample     (default: 42; must match the run)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/transfer}"
SEED="${SEED:-42}"

if [ -z "$PYTHON" ]; then
  echo "ERROR: no python interpreter found. Install Python 3.9+ or set PYTHON=/path/to/python." >&2
  exit 1
fi

if ! ls "$OUTPUT_DIR"/y_prob_*.npy >/dev/null 2>&1; then
  echo "ERROR: no saved predictions (y_prob_*.npy) in $OUTPUT_DIR" >&2
  echo "       Run the full pipeline first:  bash scripts/run_full_pipeline.sh" >&2
  exit 1
fi

"$PYTHON" compute_statistics.py --results-dir "$OUTPUT_DIR" --seed "$SEED"
