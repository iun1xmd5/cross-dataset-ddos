#!/usr/bin/env bash
# =============================================================================
# Full pipeline: download -> prepare -> transfer experiment -> statistics
#
#   1. scripts/download_datasets.py       fetch CICDDoS2019 + CICIDS2017 (kagglehub)
#   2. scripts/prepare_cicddos2019.py     clean + cap CICDDoS2019
#   3. scripts/prepare_cicids2017.py      clean + cap CICIDS2017
#   4. cross_dataset_transfer.py          train on one dataset, test on the other
#   5. scripts/run_statistics_only.sh     recompute metrics from saved predictions
#
# Usage:
#   bash scripts/run_full_pipeline.sh [--skip-download] [--use-coral] [--force] [-h]
#
# Options (or environment variables of the same meaning):
#   --skip-download   don't download; use CICDDOS_RAW / CICIDS_RAW or the existing manifest
#   --use-coral       enable CORAL feature alignment (USE_CORAL=1). Default is the pure
#                     zero-shot baseline (no alignment).
#   --force           rebuild the processed datasets even if they are up to date
#
# Environment variables (all optional):
#   PYTHON          python interpreter                       (default: python3 / python)
#   DATA_DIR        raw + processed data root                (default: <repo>/data)
#   OUTPUT_DIR      results directory                        (default: <repo>/results/transfer)
#   CICDDOS_RAW     existing raw CICDDoS2019 directory       (skips the manifest lookup)
#   CICIDS_RAW      existing raw CICIDS2017 directory        (skips the manifest lookup)
#   KAGGLEHUB_CACHE kagglehub cache location (put it on a big disk; CICDDoS2019 is huge)
#   PER_CLASS_CAP   max rows kept per class label            (default: 15000)
#   SEED            random seed                              (default: 42)
#   FEATURE_MODE    auto | hardcoded                         (default: auto)
#
# Run with `bash` (not ./) so it works on Windows/Git Bash and on GitHub downloads
# where the executable bit is not preserved.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/transfer}"
PER_CLASS_CAP="${PER_CLASS_CAP:-15000}"
SEED="${SEED:-42}"
FEATURE_MODE="${FEATURE_MODE:-auto}"
USE_CORAL="${USE_CORAL:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
CICDDOS_RAW="${CICDDOS_RAW:-}"
CICIDS_RAW="${CICIDS_RAW:-}"

for arg in "$@"; do
  case "$arg" in
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --use-coral)     USE_CORAL=1 ;;
    --force)         FORCE_PREPARE=1 ;;
    -h|--help)       sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

if [ -z "$PYTHON" ]; then
  echo "ERROR: no python interpreter found. Install Python 3.9+ or set PYTHON=/path/to/python." >&2
  exit 1
fi

step() { printf '\n=== %s ===\n' "$1"; }

MANIFEST="$DATA_DIR/raw/manifest.json"
PROCESSED_DDOS="$DATA_DIR/processed/cicddos2019"
PROCESSED_IDS="$DATA_DIR/processed/cicids2017"

# -- 1. download --------------------------------------------------------------
if [ "$SKIP_DOWNLOAD" = "1" ]; then
  step "1/5 Download (skipped)"
else
  step "1/5 Downloading datasets"
  "$PYTHON" scripts/download_datasets.py --output-dir "$DATA_DIR/raw"
fi

# -- 2 & 3. prepare -----------------------------------------------------------
PREP_FLAGS=(--per-class-cap "$PER_CLASS_CAP" --manifest "$MANIFEST")
[ "$FORCE_PREPARE" = "1" ] && PREP_FLAGS+=(--force)

step "2/5 Preparing CICDDoS2019"
DDOS_INPUT=()
[ -n "$CICDDOS_RAW" ] && DDOS_INPUT=(--input "$CICDDOS_RAW")
"$PYTHON" scripts/prepare_cicddos2019.py "${PREP_FLAGS[@]}" ${DDOS_INPUT[@]+"${DDOS_INPUT[@]}"} \
  --output "$PROCESSED_DDOS"

step "3/5 Preparing CICIDS2017"
IDS_INPUT=()
[ -n "$CICIDS_RAW" ] && IDS_INPUT=(--input "$CICIDS_RAW")
"$PYTHON" scripts/prepare_cicids2017.py "${PREP_FLAGS[@]}" ${IDS_INPUT[@]+"${IDS_INPUT[@]}"} \
  --output "$PROCESSED_IDS"

# -- 4. experiment ------------------------------------------------------------
step "4/5 Running zero-shot transfer experiment"
EXP_FLAGS=()
[ "$USE_CORAL" = "1" ] && EXP_FLAGS+=(--use-coral)
"$PYTHON" cross_dataset_transfer.py \
  --cicddos-path "$PROCESSED_DDOS" \
  --cicids-path "$PROCESSED_IDS" \
  --output "$OUTPUT_DIR" \
  --feature-mode "$FEATURE_MODE" \
  --per-class-cap "$PER_CLASS_CAP" \
  --seed "$SEED" \
  ${EXP_FLAGS[@]+"${EXP_FLAGS[@]}"}

# -- 5. statistics ------------------------------------------------------------
step "5/5 Recomputing statistics"
PYTHON="$PYTHON" OUTPUT_DIR="$OUTPUT_DIR" SEED="$SEED" bash "$ROOT/scripts/run_statistics_only.sh"

step "Done"
echo "Results: $OUTPUT_DIR"
