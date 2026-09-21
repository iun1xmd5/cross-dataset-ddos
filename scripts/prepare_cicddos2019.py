#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare CICDDoS2019 for the cross-dataset transfer experiment.

What it does (identical row selection to the notebook / cross_dataset_transfer.py):
  1. Streams every ``*.csv`` under the raw dataset directory (sorted order) in chunks.
  2. Strips / normalises column names and keeps only the 26 flow features used by
     the experiment, plus the label column.
  3. Replaces +/-inf with NaN and drops rows with missing values.
  4. Upper-cases labels and keeps BENIGN plus every DDoS attack class
     (DrDoS_*, LDAP, MSSQL, NetBIOS, Portmap, Syn, TFTP, UDP, UDP-lag, WebDDoS).
  5. Keeps the first ``--per-class-cap`` rows of each class.
  6. Writes ``<output>/cicddos2019_processed.csv`` and ``<output>/summary.json``.

The processed folder can be passed straight to ``cross_dataset_transfer.py
--cicddos-path``. The multi-class label is preserved; the experiment collapses it
to BENIGN vs ATTACK itself.

Examples
--------
    python scripts/prepare_cicddos2019.py                       # uses data/raw/manifest.json
    python scripts/prepare_cicddos2019.py --input /data/cicddos2019 --per-class-cap 15000
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_KEY = "cicddos2019"

# The 26 canonical features shared by both datasets (same names as the notebook).
CANONICAL_FEATURES: List[str] = [
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
    "Fwd Header Length.1",
    "Subflow Fwd Bytes",
    "Bwd Packets/s",
    "Bwd Packet Length Mean",
    "Init_Win_bytes_backward",
]
LABEL_CANDIDATES = ["Label", "label", "class", "Class", "attack_type"]
OUT_LABEL = "Label"

# Labels treated as attacks (after upper-casing). BENIGN is the negative class.
ATTACK_EXACT = {
    "LDAP", "MSSQL", "NETBIOS", "PORTMAP", "SYN", "TFTP",
    "UDP", "UDP-LAG", "UDPLAG", "WEBDDOS", "DDOS",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_cicddos2019")


def _key(name: str) -> str:
    """Case/space/punctuation-insensitive column key."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_FEATURE_BY_KEY = {_key(f): f for f in CANONICAL_FEATURES}
_LABEL_KEYS = {_key(c) for c in LABEL_CANDIDATES}


def keep_label(label: str) -> bool:
    return label == "BENIGN" or label.startswith("DRDOS") or label in ATTACK_EXACT


def resolve_input(explicit: Optional[Path], manifest: Path) -> Path:
    if explicit is not None:
        return Path(explicit)
    if manifest.exists():
        entry = json.loads(manifest.read_text(encoding="utf-8")).get(DATASET_KEY)
        if entry:
            return Path(entry["path"])
    fallback = REPO_ROOT / "data" / "raw" / DATASET_KEY
    if fallback.exists():
        return fallback
    raise SystemExit(
        f"Could not locate raw {DATASET_KEY}. Run scripts/download_datasets.py first "
        f"or pass --input <dir with the CSV files>."
    )


def _usecols(col: str) -> bool:
    k = _key(col)
    return k in _FEATURE_BY_KEY or k in _LABEL_KEYS


def build(input_dir: Path, per_class_cap: int, chunk_size: int) -> Dict:
    csv_paths = sorted(input_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")

    log.info("Streaming %d CSV files from %s", len(csv_paths), input_dir)
    buckets: Dict[str, List[pd.DataFrame]] = {}
    counts: Dict[str, int] = {}
    discarded: Counter = Counter()
    warned_missing = False

    for path in csv_paths:
        log.info("  reading %s", path.relative_to(input_dir))
        for chunk in pd.read_csv(
            path,
            chunksize=chunk_size,
            usecols=_usecols,          # parse only the columns we need (big speed-up)
            low_memory=False,
            on_bad_lines="skip",
            encoding_errors="replace",  # tolerate stray non-UTF-8 bytes
        ):
            # Normalise column names -> canonical feature names / "Label"
            rename = {}
            for col in chunk.columns:
                k = _key(col)
                if k in _FEATURE_BY_KEY:
                    rename[col] = _FEATURE_BY_KEY[k]
                elif k in _LABEL_KEYS:
                    rename[col] = OUT_LABEL
            chunk = chunk.rename(columns=rename)

            if OUT_LABEL not in chunk.columns:
                raise KeyError(f"No label column found in {path.name}: {list(chunk.columns)}")

            missing = [f for f in CANONICAL_FEATURES if f not in chunk.columns]
            if missing and not warned_missing:
                log.warning("%s is missing features: %s", path.name, missing)
                warned_missing = True

            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk[OUT_LABEL] = chunk[OUT_LABEL].astype(str).str.strip().str.upper()

            mask = chunk[OUT_LABEL].map(keep_label)
            if (~mask).any():
                discarded.update(chunk.loc[~mask, OUT_LABEL].value_counts().to_dict())
            chunk = chunk[mask]
            if chunk.empty:
                continue

            cols = [c for c in CANONICAL_FEATURES if c in chunk.columns] + [OUT_LABEL]
            chunk = chunk[cols].dropna()

            for label, group in chunk.groupby(OUT_LABEL):
                room = per_class_cap - counts.get(label, 0)
                if room <= 0:
                    continue
                take = group.head(room)
                buckets.setdefault(label, []).append(take)
                counts[label] = counts.get(label, 0) + len(take)

    if not buckets:
        raise RuntimeError("No rows survived filtering - check the input directory")

    df = pd.concat([f for frames in buckets.values() for f in frames], ignore_index=True)
    return {"df": df, "counts": counts, "discarded": dict(discarded)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=None,
                        help="raw CICDDoS2019 directory (default: from data/raw/manifest.json)")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "raw" / "manifest.json",
                        help="manifest written by download_datasets.py")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "processed" / DATASET_KEY,
                        help="output directory (default: <repo>/data/processed/cicddos2019)")
    parser.add_argument("--per-class-cap", type=int, default=15_000,
                        help="max rows kept per class label (default: 15000)")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if an up-to-date output already exists")
    args = parser.parse_args()

    input_dir = resolve_input(args.input, args.manifest)
    out_dir: Path = args.output
    out_csv = out_dir / f"{DATASET_KEY}_processed.csv"
    out_summary = out_dir / "summary.json"

    # Skip if an identical build already exists.
    if out_csv.exists() and out_summary.exists() and not args.force:
        try:
            prev = json.loads(out_summary.read_text(encoding="utf-8"))
            if prev.get("per_class_cap") == args.per_class_cap and prev.get("input") == str(input_dir):
                log.info("Up-to-date output found at %s - skipping (use --force to rebuild)", out_csv)
                return 0
        except json.JSONDecodeError:
            pass

    t0 = time.time()
    res = build(input_dir, args.per_class_cap, args.chunk_size)
    df: pd.DataFrame = res["df"]

    # The experiment casts every feature to float32 immediately, so store float32 values
    # with 9 significant digits (exact float32 round-trip). This keeps the file small and
    # makes the processed data bit-identical to what the notebook builds from the raw CSVs.
    # Rows are ordered by label (original order kept inside each label) so the file is
    # deterministic and does not depend on chunk boundaries when it is read back.
    df = df.sort_values(OUT_LABEL, kind="stable").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != OUT_LABEL]
    df[feature_cols] = df[feature_cols].astype("float32")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.9g")

    n_benign = int((df[OUT_LABEL] == "BENIGN").sum())
    summary = {
        "dataset": DATASET_KEY,
        "input": str(input_dir),
        "output_csv": str(out_csv),
        "per_class_cap": args.per_class_cap,
        "n_rows": int(len(df)),
        "n_benign": n_benign,
        "n_attack": int(len(df) - n_benign),
        "features": [c for c in df.columns if c != OUT_LABEL],
        "kept_per_class": dict(sorted(res["counts"].items(), key=lambda kv: -kv[1])),
        "discarded_labels": res["discarded"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("Wrote %s (%d rows, %d features)", out_csv, len(df), df.shape[1] - 1)
    log.info("  BENIGN=%d  ATTACK=%d", n_benign, len(df) - n_benign)
    for label, cnt in summary["kept_per_class"].items():
        log.info("    %-20s %7d", label, cnt)
    if res["discarded"]:
        log.info("  Discarded (non-DDoS) labels: %s", res["discarded"])
    log.info("Done in %.1f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
