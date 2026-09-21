#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the two benchmark datasets used by the transfer experiment.

    cicddos2019 -> rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files
    cicids2017  -> chethuhn/network-intrusion-dataset

Datasets are fetched with `kagglehub` (the same mechanism the notebook uses).
kagglehub stores them in its own cache directory; this script records where
each dataset ended up in ``<output-dir>/manifest.json`` so the ``prepare_*.py``
scripts can find them without copying any data.

Kaggle credentials are required for most setups: either run
``python -c "import kagglehub; kagglehub.login()"`` once, or export
KAGGLE_USERNAME and KAGGLE_KEY.

WARNING: the CICDDoS2019 download is very large (tens of GB). Use
``--cache-dir`` to put the kagglehub cache on a disk with enough space.

Examples
--------
    python scripts/download_datasets.py                     # both datasets
    python scripts/download_datasets.py --dataset cicids2017
    python scripts/download_datasets.py --cache-dir /mnt/bigdisk/kagglehub
    python scripts/download_datasets.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS: Dict[str, str] = {
    "cicddos2019": "rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files",
    "cicids2017": "chethuhn/network-intrusion-dataset",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download")


def _csv_stats(path: Path) -> Dict[str, float]:
    csvs = sorted(path.rglob("*.csv"))
    total = sum(p.stat().st_size for p in csvs)
    return {"n_csv": len(csvs), "size_gb": round(total / 1024 ** 3, 3)}


def _load_manifest(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Ignoring unreadable manifest %s", path)
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS],
        default="all",
        help="which dataset to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="where manifest.json is written (default: <repo>/data/raw)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="kagglehub cache directory (default: kagglehub's own default, "
        "usually ~/.cache/kagglehub)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if already cached"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would be downloaded and exit"
    )
    args = parser.parse_args()

    selected: List[str] = list(DATASETS) if args.dataset == "all" else [args.dataset]

    if args.dry_run:
        for name in selected:
            log.info("[dry-run] would download %s from kaggle: %s", name, DATASETS[name])
        return 0

    # Must be set before kagglehub resolves its cache folder.
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = str(args.cache_dir.resolve())
        log.info("kagglehub cache: %s", os.environ["KAGGLEHUB_CACHE"])

    try:
        import kagglehub
    except ImportError:
        log.error("kagglehub is not installed. Run: pip install kagglehub")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)

    failures = 0
    for name in selected:
        slug = DATASETS[name]
        log.info("Downloading %s (%s) ...", name, slug)
        t0 = time.time()
        try:
            local_path = Path(kagglehub.dataset_download(slug, force_download=args.force))
        except Exception as exc:  # network, auth, quota ...
            failures += 1
            log.error("Failed to download %s: %s", name, exc)
            log.error(
                "If this is an authentication problem, run "
                "`python -c \"import kagglehub; kagglehub.login()\"` or set "
                "KAGGLE_USERNAME / KAGGLE_KEY, then retry."
            )
            continue

        stats = _csv_stats(local_path)
        if stats["n_csv"] == 0:
            failures += 1
            log.error("%s downloaded to %s but no CSV files were found", name, local_path)
            continue

        manifest[name] = {
            "slug": slug,
            "path": str(local_path),
            **stats,
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log.info(
            "%s ready: %d CSV files, %.2f GB, at %s (%.0f s)",
            name, stats["n_csv"], stats["size_gb"], local_path, time.time() - t0,
        )

    if failures:
        log.error("%d dataset(s) failed", failures)
        return 1

    log.info("Manifest written to %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
