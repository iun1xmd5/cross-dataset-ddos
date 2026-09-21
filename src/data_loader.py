#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset loading for the transfer experiment.

  * resolve_dataset_path   - explicit folder, or download through kagglehub
  * load_dataset           - stream CSVs in chunks, normalise column names, cap rows per class
  * aggregate_binary       - collapse multi-class labels into BENIGN=0 / ATTACK=1
  * load_transfer_datasets - convenience: load + binarise CICDDoS2019 and CICIDS2017

Folders produced by scripts/prepare_*.py can be passed here directly; so can the raw
Kaggle folders.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    CICDDOS2019_KAGGLE_SLUG,
    CICIDS2017_KAGGLE_SLUG,
    Config,
    log,
)

# Raw column name (lower-case, spaces/slashes/dashes -> "_") -> canonical feature name.
COLUMN_ALIASES = {
    "avg_packet_size": "Average Packet Size",
    "average_packet_size": "Average Packet Size",
    "packet_length_mean": "Packet Length Mean",
    "flow_byts_s": "Flow Bytes/s",
    "flow_bytes_s": "Flow Bytes/s",
    "flow_iat_mean": "Flow IAT Mean",
    "fwd_packets_s": "Fwd Packets/s",
    "init_win_bytes_forward": "Init_Win_bytes_forward",
    "flow_duration": "Flow Duration",
    "flow_packets_s": "Flow Packets/s",
    "max_packet_length": "Max Packet Length",
    "min_packet_length": "Min Packet Length",
    "packet_length_std": "Packet Length Std",
    "packet_length_variance": "Packet Length Variance",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_total": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "fwd_header_length": "Fwd Header Length",
    "fwd_header_length_1": "Fwd Header Length.1",
    "subflow_fwd_bytes": "Subflow Fwd Bytes",
    "bwd_packets_s": "Bwd Packets/s",
    "bwd_packet_length_mean": "Bwd Packet Length Mean",
    "init_win_bytes_backward": "Init_Win_bytes_backward",
}

LABEL_CANDIDATES = ["Label", "label", "class", "Class", "attack_type"]

# Labels (upper-cased) that count as ATTACK. Everything else except BENIGN is dropped.
ATTACK_EXACT = {
    "LDAP", "MSSQL", "NETBIOS", "PORTMAP", "SYN", "TFTP",
    "UDP", "UDP-LAG", "UDPLAG", "WEBDDOS", "DDOS",
}


def resolve_dataset_path(explicit_path: Optional[Path], kaggle_slug: str, label: str) -> Path:
    """Use the given folder, or download the dataset with kagglehub."""
    if explicit_path is not None:
        return Path(explicit_path)
    try:
        import kagglehub
    except Exception as exc:
        raise RuntimeError(
            f"No path given for {label} and kagglehub is not installed. "
            "Install with `pip install kagglehub` or pass the path explicitly."
        ) from exc
    log.info("No path given for %s — downloading via kagglehub (%s)", label, kaggle_slug)
    path = Path(kagglehub.dataset_download(kaggle_slug))
    log.info("%s downloaded to: %s", label, path)
    return path


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        key = col.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)


def _detect_label_column(df: pd.DataFrame) -> str:
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
    raise KeyError(f"No label column found in {list(df.columns)}")


def load_dataset(
    data_dir: Path,
    per_class_cap: int,
    chunk_size: int,
    allowed_classes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Stream every CSV under ``data_dir`` and keep the first ``per_class_cap`` rows per label."""
    data_dir = Path(data_dir)
    csv_paths = sorted(data_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs under {data_dir}")

    log.info("Loading %d CSV files from %s", len(csv_paths), data_dir)
    buckets: Dict[str, List[pd.DataFrame]] = {}
    counts: Dict[str, int] = {}

    for path in csv_paths:
        log.info("  streaming %s", path.name)
        for chunk in pd.read_csv(
            path, chunksize=chunk_size, low_memory=False, on_bad_lines="skip",
        ):
            chunk = _normalize_columns(chunk)
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            label_col = _detect_label_column(chunk)
            chunk[label_col] = chunk[label_col].astype(str).str.strip().str.upper()

            if allowed_classes is not None:
                chunk = chunk[chunk[label_col].isin(allowed_classes)]
                if chunk.empty:
                    continue

            keep_cols = [
                c for c in chunk.columns
                if c == label_col or c in COLUMN_ALIASES.values()
            ]
            chunk = chunk[keep_cols].dropna()

            for label, group in chunk.groupby(label_col):
                room = per_class_cap - counts.get(label, 0)
                if room <= 0:
                    continue
                take = group.head(room)
                buckets.setdefault(label, []).append(take)
                counts[label] = counts.get(label, 0) + len(take)

    if not buckets:
        raise RuntimeError("No rows survived filtering")

    df = pd.concat([c for frames in buckets.values() for c in frames], ignore_index=True)
    log.info("Loaded %d rows across %d classes", len(df), len(counts))
    for label, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        log.info("    %-25s %7d", label, cnt)
    return df


def aggregate_binary(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Aggregate multi-class labels into {BENIGN=0, ATTACK=1}; other classes are dropped."""
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()

    def to_binary(lbl: str) -> Optional[str]:
        if lbl == "BENIGN":
            return "BENIGN"
        if lbl.startswith("DRDOS") or lbl in ATTACK_EXACT:
            return "ATTACK"
        return None

    df["__binary_label__"] = df[label_col].map(to_binary)
    df = df[df["__binary_label__"].notna()].copy()
    df["__binary_label__"] = (df["__binary_label__"] == "ATTACK").astype(int)
    return df


def load_transfer_datasets(
    cicddos_path: Optional[Path],
    cicids_path: Optional[Path],
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both datasets and add the binary label column ``__binary_label__``."""
    ddos_dir = resolve_dataset_path(cicddos_path, CICDDOS2019_KAGGLE_SLUG, "CICDDoS2019")
    ids_dir = resolve_dataset_path(cicids_path, CICIDS2017_KAGGLE_SLUG, "CICIDS2017")

    cicddos = load_dataset(ddos_dir, cfg.per_class_cap, cfg.chunk_size)
    cicids = load_dataset(ids_dir, cfg.per_class_cap, cfg.chunk_size)

    cicddos = aggregate_binary(cicddos, _detect_label_column(cicddos))
    cicids = aggregate_binary(cicids, _detect_label_column(cicids))

    for name, df in (("CICDDoS2019", cicddos), ("CICIDS2017 ", cicids)):
        log.info(
            "Aggregated %s: %d rows (BENIGN=%d, ATTACK=%d)",
            name, len(df),
            int((df["__binary_label__"] == 0).sum()),
            int((df["__binary_label__"] == 1).sum()),
        )
    return cicddos, cicids
