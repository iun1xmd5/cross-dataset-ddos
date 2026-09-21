#!/usr/bin/env bash
# =============================================================================
# Statistics only: recompute every metric and run the significance tests from the
# predictions saved by a previous run (y_prob_*.npy + y_true_*.npy). No datasets are
# read and no model is trained, so this needs no GPU and no TensorFlow.
#
#   1. src/evaluate.py       ROC-AUC, accuracy / balanced accuracy / macro-F1 at 0.5 and at the
#                            optimal threshold, confusion matrices, score-inversion diagnostic
#                            -> transfer_results_recomputed.csv, transfer_statistics.json
#   2. src/significance.py   bootstrap confidence intervals, McNemar's test and paired
#                            bootstrap of the AUC difference (Holm-adjusted)
#                            -> significance_confidence_intervals.csv, significance_pairwise.csv
#
# Usage:
#   bash scripts/run_statistics_only.sh
#   OUTPUT_DIR=/kaggle/working/results/transfer N_BOOT=2000 bash scripts/run_statistics_only.sh
#
# Environment variables (all optional):
#   PYTHON       python interpreter                       (default: python3 / python)
#   OUTPUT_DIR   directory holding y_prob_*/y_true_*.npy   (default: <repo>/results/transfer)
#   SEED         seed of the balanced subsample / bootstrap (default: 42; match the run's seed)
#   N_BOOT       bootstrap resamples                       (default: 1000)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/transfer}"
SEED="${SEED:-42}"
N_BOOT="${N_BOOT:-1000}"

if [ -z "$PYTHON" ]; then
  echo "ERROR: no python interpreter found. Install Python 3.9+ or set PYTHON=/path/to/python." >&2
  exit 1
fi

if ! ls "$OUTPUT_DIR"/y_prob_*.npy >/dev/null 2>&1; then
  echo "ERROR: no saved predictions (y_prob_*.npy) in $OUTPUT_DIR" >&2
  echo "       Run the full pipeline first:  bash scripts/run_full_pipeline.sh" >&2
  exit 1
fi

echo "=== Metrics ==="
"$PYTHON" src/evaluate.py --results-dir "$OUTPUT_DIR" --seed "$SEED"

echo
echo "=== Significance tests ==="
"$PYTHON" src/significance.py --results-dir "$OUTPUT_DIR" --n-boot "$N_BOOT" --seed "$SEED"
