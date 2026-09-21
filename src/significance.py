#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical significance of model differences on the target dataset.

Works purely from the predictions saved by train.py (``y_prob_*.npy`` / ``y_true_*.npy``);
no datasets are read and nothing is retrained.

What is computed, per transfer direction
  1. Bootstrap confidence intervals (percentile method) for every model: ROC-AUC,
     accuracy @0.5, accuracy @optimal threshold and macro-F1 @0.5.
  2. Pairwise comparisons between models evaluated on the SAME target samples:
       * McNemar's exact test on per-sample correctness (at the fixed 0.5 threshold, and at
         each model's own optimal threshold)
       * paired bootstrap of the ROC-AUC difference (confidence interval + p-value)
     p-values are Holm-adjusted within each family of comparisons.

Which models can be compared?
  Neural models predict one label per WINDOW of flows, tree models one label per FLOW, so
  their predictions do not refer to the same samples and cannot be paired. Models are
  therefore only compared when their ``y_true`` arrays are identical: LSTM / BiLSTM /
  CNN-LSTM with each other, and Random Forest with XGBoost.

Caveats
  * Consecutive network flows are correlated, so the bootstrap treats them as exchangeable
    and the intervals are optimistic (too narrow). Read borderline p-values with care.
  * The "optimal" threshold is chosen on the same target data it is evaluated on (Youden's
    J), which favours every model; it is reported for completeness. Prefer ROC-AUC and the
    0.5-threshold results for claims about zero-shot transfer.

Usage
    python src/significance.py --results-dir results/transfer --n-boot 1000

Output
    <results-dir>/significance_confidence_intervals.csv
    <results-dir>/significance_pairwise.csv
"""

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from config import ALL_MODELS, NEURAL_MODELS, TREE_MODELS, log
from evaluate import discover_predictions, find_optimal_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def macro_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    f1_attack = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_benign = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return (f1_attack + f1_benign) / 2.0


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> Tuple[int, int, float]:
    """Exact (binomial) McNemar test. Returns (#only A correct, #only B correct, p-value)."""
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return only_a, only_b, p


def bootstrap_p_value(diff: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0 (with +1 smoothing)."""
    b = len(diff)
    tail = min(int(np.sum(diff <= 0)), int(np.sum(diff >= 0)))
    return min(1.0, 2.0 * (tail + 1) / (b + 1))


def group_by_target(preds: List[Tuple[str, str, str, np.ndarray, np.ndarray]]):
    """
    Group saved predictions into comparable sets: same direction AND identical y_true.
    Returns {(source, target, group_id): {"y_true": array, "models": {model: y_prob}}}.
    """
    groups: Dict[Tuple[str, str, int], Dict] = {}
    for src, tgt, model, y_true, y_prob in preds:
        placed = False
        for (s, t, gid), g in groups.items():
            if (s, t) == (src, tgt) and len(g["y_true"]) == len(y_true) and np.array_equal(g["y_true"], y_true):
                g["models"][model] = y_prob
                placed = True
                break
        if not placed:
            gid = sum(1 for (s, t, _) in groups if (s, t) == (src, tgt))
            groups[(src, tgt, gid)] = {"y_true": y_true, "models": {model: y_prob}}
    return groups


def group_label(models: Sequence[str]) -> str:
    if all(m in NEURAL_MODELS for m in models):
        return "windows (neural)"
    if all(m in TREE_MODELS for m in models):
        return "flows (trees)"
    return "mixed"


# ---------------------------------------------------------------------------
# Core analysis for one comparable group
# ---------------------------------------------------------------------------
def analyse_group(
    direction: str,
    y_true: np.ndarray,
    probs: Dict[str, np.ndarray],
    n_boot: int,
    alpha: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    models = [m for m in ALL_MODELS if m in probs]
    label = group_label(models)
    n = len(y_true)
    y_true = np.asarray(y_true).astype(int)
    thr_opt = {m: find_optimal_threshold(y_true, probs[m])[0] for m in models}

    # ---- point estimates on the full target set --------------------------------------
    point: Dict[str, Dict[str, float]] = {}
    for m in models:
        p = probs[m]
        try:
            auc = float(roc_auc_score(y_true, p))
        except ValueError:
            auc = float("nan")
        point[m] = {
            "auc": auc,
            "acc_0.5": float(np.mean((p >= 0.5).astype(int) == y_true)),
            "acc_opt": float(np.mean((p >= thr_opt[m]).astype(int) == y_true)),
            "macro_f1_0.5": macro_f1(y_true, (p >= 0.5).astype(int)),
        }

    # ---- bootstrap (the SAME resamples for every model in the group -> paired) --------
    rng = np.random.RandomState(seed)
    keys = ("auc", "acc_0.5", "acc_opt", "macro_f1_0.5")
    boot = {m: {k: [] for k in keys} for m in models}
    done = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():          # resample with a single class: AUC undefined
            continue
        for m in models:
            pb = probs[m][idx]
            boot[m]["auc"].append(roc_auc_score(yt, pb))
            boot[m]["acc_0.5"].append(np.mean((pb >= 0.5).astype(int) == yt))
            boot[m]["acc_opt"].append(np.mean((pb >= thr_opt[m]).astype(int) == yt))
            boot[m]["macro_f1_0.5"].append(macro_f1(yt, (pb >= 0.5).astype(int)))
        done += 1
    boot = {m: {k: np.asarray(v) for k, v in d.items()} for m, d in boot.items()}
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)

    ci_rows: List[Dict] = []
    for m in models:
        row = {"direction": direction, "group": label, "model": m, "n": n, "n_boot": done}
        for k in keys:
            row[k] = point[m][k]
            row[f"{k}_lo"] = float(np.percentile(boot[m][k], lo_q)) if done else float("nan")
            row[f"{k}_hi"] = float(np.percentile(boot[m][k], hi_q)) if done else float("nan")
        ci_rows.append(row)

    # ---- pairwise tests ---------------------------------------------------------------
    pair_rows: List[Dict] = []
    for a, b in itertools.combinations(models, 2):
        correct = {
            "0.5": ((probs[a] >= 0.5).astype(int) == y_true, (probs[b] >= 0.5).astype(int) == y_true),
            "opt": ((probs[a] >= thr_opt[a]).astype(int) == y_true, (probs[b] >= thr_opt[b]).astype(int) == y_true),
        }
        only_a, only_b, p05 = mcnemar_exact(*correct["0.5"])
        _, _, p_opt = mcnemar_exact(*correct["opt"])

        d_auc = boot[a]["auc"] - boot[b]["auc"]
        pair_rows.append({
            "direction": direction, "group": label, "model_a": a, "model_b": b, "n": n,
            "acc_a": point[a]["acc_0.5"], "acc_b": point[b]["acc_0.5"],
            "delta_acc": point[a]["acc_0.5"] - point[b]["acc_0.5"],
            "only_a_correct": only_a, "only_b_correct": only_b,
            "mcnemar_p_0.5": p05,
            "mcnemar_p_opt": p_opt,
            "auc_a": point[a]["auc"], "auc_b": point[b]["auc"],
            "delta_auc": point[a]["auc"] - point[b]["auc"],
            "delta_auc_lo": float(np.percentile(d_auc, lo_q)) if done else float("nan"),
            "delta_auc_hi": float(np.percentile(d_auc, hi_q)) if done else float("nan"),
            "delta_auc_p": bootstrap_p_value(d_auc) if done else float("nan"),
        })

    # Holm correction within this family of comparisons
    if pair_rows:
        for col in ("mcnemar_p_0.5", "mcnemar_p_opt", "delta_auc_p"):
            adj = holm_adjust([r[col] for r in pair_rows])
            for r, v in zip(pair_rows, adj):
                r[f"{col}_holm"] = v
        for r in pair_rows:
            r["significant_mcnemar_0.5"] = bool(r["mcnemar_p_0.5_holm"] < alpha)
            r["significant_auc"] = bool(r["delta_auc_p_holm"] < alpha)
    return ci_rows, pair_rows


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/transfer"))
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples (default: 1000)")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        log.error("Results directory not found: %s", args.results_dir)
        return 1
    preds = discover_predictions(args.results_dir)
    if not preds:
        log.error("No y_prob_*.npy / y_true_*.npy pairs in %s. Run the experiment first.", args.results_dir)
        return 1

    groups = group_by_target(preds)
    ci_all: List[Dict] = []
    pair_all: List[Dict] = []
    for (src, tgt, _), g in sorted(groups.items()):
        direction = f"{src}->{tgt}"
        models = [m for m in ALL_MODELS if m in g["models"]]
        log.info("%s | %s | n=%d | models: %s | %d bootstrap resamples",
                 direction, group_label(models), len(g["y_true"]), ", ".join(models), args.n_boot)
        ci_rows, pair_rows = analyse_group(direction, g["y_true"], g["models"],
                                           args.n_boot, args.alpha, args.seed)
        ci_all += ci_rows
        pair_all += pair_rows

    ci_df = pd.DataFrame(ci_all)
    pair_df = pd.DataFrame(pair_all)
    ci_path = args.results_dir / "significance_confidence_intervals.csv"
    pair_path = args.results_dir / "significance_pairwise.csv"
    ci_df.to_csv(ci_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    log.info("Wrote %s", ci_path)
    log.info("Wrote %s", pair_path)

    pct = int(round(100 * (1 - args.alpha)))
    print(f"\n{pct}% bootstrap confidence intervals ({args.n_boot} resamples)")
    show = ci_df.copy()
    for k in ("auc", "acc_0.5", "acc_opt", "macro_f1_0.5"):
        show[k] = show.apply(lambda r, k=k: f"{r[k]:.4f} [{r[k + '_lo']:.4f}, {r[k + '_hi']:.4f}]", axis=1)
    print(show[["direction", "group", "model", "n", "auc", "acc_0.5", "acc_opt", "macro_f1_0.5"]]
          .to_string(index=False))

    if pair_df.empty:
        print("\nNo comparable model pairs (need >= 2 models evaluated on the same target samples).")
    else:
        print(f"\nPairwise tests (Holm-adjusted p-values; significant if p < {args.alpha})")
        cols = ["direction", "model_a", "model_b", "delta_acc", "mcnemar_p_0.5_holm",
                "delta_auc", "delta_auc_lo", "delta_auc_hi", "delta_auc_p_holm",
                "significant_mcnemar_0.5", "significant_auc"]
        print(pair_df[cols].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
