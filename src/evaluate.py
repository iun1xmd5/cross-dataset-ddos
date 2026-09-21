#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation, diagnostics and reporting.

Library functions (used by train.py, sequence_length_sweep.py, computational_cost.py)
  * evaluate_full        - ROC-AUC, accuracy / balanced accuracy / macro-F1 at the fixed 0.5
                           threshold and at the Youden-optimal threshold, confusion matrices,
                           probability summaries and the score-inversion diagnostic
  * balanced_metrics     - the same metrics on a class-balanced subsample of the target
  * record_result        - evaluate one model, store metrics and save y_prob / y_true (.npy)
  * write_results_csv, print_paper_table - reporting

Command line (no retraining, no datasets needed)
    python src/evaluate.py --results-dir results/transfer

recomputes every metric from the saved ``y_prob_*.npy`` / ``y_true_*.npy`` files and writes
    <results-dir>/transfer_results_recomputed.csv
    <results-dir>/transfer_statistics.json
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import ALL_MODELS, MODEL_TAGS, TAG_TO_MODEL, log
from preprocessing import balanced_indices

BAL_KEYS = (
    "roc_auc", "acc_0.5", "bal_acc_0.5", "macro_f1_0.5",
    "acc_opt", "bal_acc_opt", "macro_f1_opt", "thr_opt",
)
TAG_RE = re.compile(r"^(cicddos2019|cicids2017)_to_(cicddos2019|cicids2017)_(.+)$")


@dataclass
class DirectionResult:
    source: str
    target: str
    shared_features: List[str]
    results: Dict[str, Dict] = field(default_factory=dict)  # model -> metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, str]:
    """Return (threshold, method) using Youden's J; fall back to F1-max."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    if len(j) > 0:
        idx = int(np.argmax(j))
        return float(thr[idx]), "youden"
    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    idx = int(np.argmax(f1s))
    t = float(thr_pr[idx]) if idx < len(thr_pr) else 0.5
    return t, "f1"


def evaluate_full(y_true: np.ndarray, y_prob: np.ndarray, prefix: str = "", verbose: bool = True) -> Dict:
    """Full diagnostic evaluation at the fixed 0.5 threshold and at the optimal threshold."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    # Mean predicted probability per true class (anti-correlation / score-inversion check)
    mean_p0 = float(y_prob[y_true == 0].mean()) if (y_true == 0).any() else float("nan")
    mean_p1 = float(y_prob[y_true == 1].mean()) if (y_true == 1).any() else float("nan")
    flipped = bool(mean_p1 < mean_p0)   # higher score on BENIGN than ATTACK -> likely inversion

    # Fixed 0.5 threshold
    y_pred05 = (y_prob >= 0.5).astype(int)
    acc05 = float(accuracy_score(y_true, y_pred05))
    bal05 = float(balanced_accuracy_score(y_true, y_pred05))
    mac05 = float(f1_score(y_true, y_pred05, average="macro", zero_division=0))
    w05 = float(f1_score(y_true, y_pred05, average="weighted", zero_division=0))
    cm05 = confusion_matrix(y_true, y_pred05, labels=[0, 1])

    # Optimal threshold
    thr_opt, thr_method = find_optimal_threshold(y_true, y_prob)
    y_pred_opt = (y_prob >= thr_opt).astype(int)
    acc_opt = float(accuracy_score(y_true, y_pred_opt))
    bal_opt = float(balanced_accuracy_score(y_true, y_pred_opt))
    mac_opt = float(f1_score(y_true, y_pred_opt, average="macro", zero_division=0))
    w_opt = float(f1_score(y_true, y_pred_opt, average="weighted", zero_division=0))
    cm_opt = confusion_matrix(y_true, y_pred_opt, labels=[0, 1])

    # Probability distribution summary
    p_mean = float(y_prob.mean())
    p_std = float(y_prob.std())
    p_p10 = float(np.percentile(y_prob, 10))
    p_p50 = float(np.percentile(y_prob, 50))
    p_p90 = float(np.percentile(y_prob, 90))

    metrics = {
        "roc_auc": auc,
        "mean_prob_benign": mean_p0,
        "mean_prob_attack": mean_p1,
        "score_flipped": flipped,
        "prob_mean": p_mean,
        "prob_std": p_std,
        "prob_p10": p_p10,
        "prob_p50": p_p50,
        "prob_p90": p_p90,
        # 0.5 threshold
        "acc_0.5": acc05,
        "bal_acc_0.5": bal05,
        "macro_f1_0.5": mac05,
        "weighted_f1_0.5": w05,
        "cm_0.5": cm05.tolist(),
        # optimal threshold
        "thr_opt": thr_opt,
        "thr_method": thr_method,
        "acc_opt": acc_opt,
        "bal_acc_opt": bal_opt,
        "macro_f1_opt": mac_opt,
        "weighted_f1_opt": w_opt,
        "cm_opt": cm_opt.tolist(),
    }

    if verbose:
        log.info(
            "%s  AUC=%.4f | meanP(benign)=%.4f meanP(attack)=%.4f  flipped=%s",
            prefix, auc, mean_p0, mean_p1, flipped,
        )
        log.info(
            "%s  @0.5  acc=%.4f bal=%.4f macroF1=%.4f  |  @opt(%.3f) acc=%.4f bal=%.4f macroF1=%.4f",
            prefix, acc05, bal05, mac05, thr_opt, acc_opt, bal_opt, mac_opt,
        )
        log.info("%s  CM@0.5=%s   CM@opt=%s", prefix, cm05.tolist(), cm_opt.tolist())
        log.info(
            "%s  prob dist: mean=%.3f std=%.3f  p10=%.3f p50=%.3f p90=%.3f",
            prefix, p_mean, p_std, p_p10, p_p50, p_p90,
        )
    return metrics


def balanced_metrics(y_true: np.ndarray, y_prob: np.ndarray, seed: int = 42, prefix: str = "",
                     verbose: bool = True) -> Optional[Dict]:
    """Evaluate on a class-balanced subsample of the target (None if it would be < 100 samples)."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    sel = balanced_indices(y_true, seed)
    if sel is None or len(sel) < 100:
        return None
    if verbose:
        log.info("  --- balanced subsample (%d samples) ---", len(sel))
    m = evaluate_full(y_true[sel], y_prob[sel], prefix=f"{prefix} [bal]", verbose=verbose)
    return {k: m[k] for k in BAL_KEYS}


def predict_keras(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X, batch_size=512, verbose=0).ravel()


def evaluate_keras(model, X, y, prefix: str = ""):
    y_prob = predict_keras(model, X)
    return evaluate_full(y, y_prob, prefix=prefix), y_prob


def evaluate_sklearn(model, X, y, prefix: str = ""):
    y_prob = model.predict_proba(X)[:, 1]
    return evaluate_full(y, y_prob, prefix=prefix), y_prob


def record_result(
    result: DirectionResult,
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    seed: int = 42,
    eval_balanced: bool = True,
) -> Dict:
    """Evaluate one model, store metrics in ``result`` and save y_prob / y_true as .npy."""
    prefix = f"  {model_name:13s}"
    metrics = evaluate_full(y_true, y_prob, prefix=prefix)
    if eval_balanced:
        bal = balanced_metrics(y_true, y_prob, seed=seed, prefix=prefix)
        if bal is not None:
            metrics["balanced_subsample"] = bal
    result.results[model_name] = metrics

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{result.source}_to_{result.target}_{MODEL_TAGS[model_name]}"
    np.save(output_dir / f"y_prob_{tag}.npy", y_prob)
    np.save(output_dir / f"y_true_{tag}.npy", y_true)
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_results_csv(
    results: List[DirectionResult],
    output_dir: Path,
    filename: str = "transfer_results_improved.csv",
) -> pd.DataFrame:
    rows = []
    for r in results:
        for model_name, m in r.results.items():
            row = {
                "source": r.source,
                "target": r.target,
                "model": model_name,
                "roc_auc": f"{m.get('roc_auc', float('nan')):.4f}",
                "mean_prob_benign": f"{m.get('mean_prob_benign', float('nan')):.4f}",
                "mean_prob_attack": f"{m.get('mean_prob_attack', float('nan')):.4f}",
                "score_flipped": m.get("score_flipped", False),
                "acc_0.5": f"{m.get('acc_0.5', float('nan')):.4f}",
                "bal_acc_0.5": f"{m.get('bal_acc_0.5', float('nan')):.4f}",
                "macro_f1_0.5": f"{m.get('macro_f1_0.5', float('nan')):.4f}",
                "thr_opt": f"{m.get('thr_opt', 0.5):.4f}",
                "acc_opt": f"{m.get('acc_opt', float('nan')):.4f}",
                "bal_acc_opt": f"{m.get('bal_acc_opt', float('nan')):.4f}",
                "macro_f1_opt": f"{m.get('macro_f1_opt', float('nan')):.4f}",
            }
            if "balanced_subsample" in m:
                b = m["balanced_subsample"]
                row["bal_sub_auc"] = f"{b.get('roc_auc', float('nan')):.4f}"
                row["bal_sub_acc_opt"] = f"{b.get('acc_opt', float('nan')):.4f}"
                row["bal_sub_macro_f1_opt"] = f"{b.get('macro_f1_opt', float('nan')):.4f}"
            rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(output_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %s", out)
    log.info("\n%s", df.to_string(index=False))
    return df


def print_paper_table(
    results: List[DirectionResult],
    title: str = "IMPROVED Table 10 — Zero-shot cross-dataset transfer (0.5 threshold vs optimal threshold)",
) -> None:
    forward = next((r for r in results if r.source.startswith("cicddos")), None)
    reverse = next((r for r in results if r.source.startswith("cicids")), None)

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "   n/a "
        return f"{v:.4f}"

    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    header = (
        f"{'Model':<16} | "
        f"{'CICDDos -> CICIDS':^46} | "
        f"{'CICIDS -> CICDDos':^46}"
    )
    print(header)
    print(
        f"{'':16} | "
        f"{'AUC':>7} {'Acc@0.5':>8} {'Acc@opt':>8} {'F1@opt':>8} | "
        f"{'AUC':>7} {'Acc@0.5':>8} {'Acc@opt':>8} {'F1@opt':>8}"
    )
    print("-" * 110)

    for model in ALL_MODELS:
        f = forward.results.get(model) if forward else None
        r = reverse.results.get(model) if reverse else None
        if not f and not r:
            continue
        print(
            f"{model:<16} | "
            f"{fmt(f['roc_auc'] if f else None):>7} "
            f"{fmt(f['acc_0.5'] if f else None):>8} "
            f"{fmt(f['acc_opt'] if f else None):>8} "
            f"{fmt(f['macro_f1_opt'] if f else None):>8} | "
            f"{fmt(r['roc_auc'] if r else None):>7} "
            f"{fmt(r['acc_0.5'] if r else None):>8} "
            f"{fmt(r['acc_opt'] if r else None):>8} "
            f"{fmt(r['macro_f1_opt'] if r else None):>8}"
        )
    print("=" * 110)

    print("\n--- Diagnostic flags ---")
    any_flag = False
    for r in results:
        for model, m in r.results.items():
            if m.get("score_flipped", False):
                any_flag = True
                print(f"  WARNING: {r.source}->{r.target} / {model}  "
                      f"meanP(attack)={m['mean_prob_attack']:.3f} < "
                      f"meanP(benign)={m['mean_prob_benign']:.3f}  → possible score inversion")
            if m.get("roc_auc", 0.5) > 0.7 and m.get("acc_0.5", 1.0) < 0.2:
                any_flag = True
                print(f"  NOTE: {r.source}->{r.target} / {model}  "
                      f"high AUC ({m['roc_auc']:.3f}) but low Acc@0.5 ({m['acc_0.5']:.3f}) "
                      f"→ threshold/calibration issue (opt thr={m['thr_opt']:.3f})")
    if not any_flag:
        print("  none")


# ---------------------------------------------------------------------------
# Saved predictions
# ---------------------------------------------------------------------------
def discover_predictions(results_dir: Path) -> List[Tuple[str, str, str, np.ndarray, np.ndarray]]:
    """Load every saved (source, target, model, y_true, y_prob) from ``results_dir``."""
    results_dir = Path(results_dir)
    found = []
    for prob_path in sorted(results_dir.glob("y_prob_*.npy")):
        tag = prob_path.stem[len("y_prob_"):]
        m = TAG_RE.match(tag)
        if not m:
            log.warning("Skipping unrecognised file %s", prob_path.name)
            continue
        true_path = results_dir / f"y_true_{tag}.npy"
        if not true_path.exists():
            log.warning("Missing %s - skipping", true_path.name)
            continue
        src, tgt, suffix = m.groups()
        model = TAG_TO_MODEL.get(suffix, suffix)
        found.append((src, tgt, model, np.load(true_path), np.load(prob_path)))
    return found


def load_saved_results(results_dir: Path, seed: int = 42) -> List[DirectionResult]:
    """Rebuild DirectionResult objects (with fresh metrics) from saved predictions."""
    results_dir = Path(results_dir)
    by_dir: Dict[Tuple[str, str], DirectionResult] = {}
    for src, tgt, model, y_true, y_prob in discover_predictions(results_dir):
        key = (src, tgt)
        if key not in by_dir:
            feat_path = results_dir / f"shared_features_{src}_to_{tgt}.json"
            feats = json.loads(feat_path.read_text()) if feat_path.exists() else []
            by_dir[key] = DirectionResult(source=src, target=tgt, shared_features=feats)
        res = by_dir[key]
        metrics = evaluate_full(y_true, y_prob, verbose=False)
        bal = balanced_metrics(y_true, y_prob, seed=seed, verbose=False)
        if bal is not None:
            metrics["balanced_subsample"] = bal
        res.results[model] = metrics
        log.info("%s -> %s | %-13s AUC=%.4f Acc@0.5=%.4f Acc@opt=%.4f",
                 src, tgt, model, metrics["roc_auc"], metrics["acc_0.5"], metrics["acc_opt"])
    return list(by_dir.values())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute all evaluation metrics from saved predictions (no retraining)."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/transfer"),
                        help="directory containing y_prob_*.npy / y_true_*.npy")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed of the balanced subsample (must match the experiment's seed)")
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        log.error("Results directory not found: %s", args.results_dir)
        return 1

    results = load_saved_results(args.results_dir, seed=args.seed)
    if not results:
        log.error("No y_prob_*.npy / y_true_*.npy pairs in %s. Run the experiment first "
                  "(scripts/run_full_pipeline.sh).", args.results_dir)
        return 1

    write_results_csv(results, args.results_dir, "transfer_results_recomputed.csv")
    stats = [
        {"source": r.source, "target": r.target, "model": model, "metrics": m}
        for r in results for model, m in r.results.items()
    ]
    json_path = args.results_dir / "transfer_statistics.json"
    json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Wrote %s", json_path)

    print_paper_table(results, title="Zero-shot cross-dataset transfer (recomputed from saved predictions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
