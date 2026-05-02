"""
Save ROC, PR, and reliability (calibration) figures for the test set.
Called from train_model.py after all models are scored.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_curve,
)


def save_all_evaluation_figures(
    y_test,
    test_scores: dict,
    best_model_name: str,
    out_dir: str,
    display_names: Optional[Dict[str, str]] = None,
) -> None:
    """
    y_test: array of 0/1
    test_scores: model_key -> P(default) on test set
    display_names: optional model_key -> short label for legend
    """
    os.makedirs(out_dir, exist_ok=True)
    y_test = np.asarray(y_test)
    leg = display_names or {}

    # ---- ROC: all models on one figure ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_score in test_scores.items():
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        label = leg.get(name, name)
        ax.plot(fpr, tpr, lw=1.5, label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — test set (all models)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    p = os.path.join(out_dir, "roc_curves_all_models.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    # ---- PR: all models ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_score in test_scores.items():
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        label = leg.get(name, name)
        ax.plot(rec, prec, lw=1.5, label=f"{label} (AP={ap:.3f})")
    base_rate = y_test.mean()
    ax.axhline(base_rate, color="k", linestyle="--", lw=0.8, label=f"baseline rate={base_rate:.3f}")
    ax.set_xlabel("Recall (default)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves — test set (imbalanced class)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    p = os.path.join(out_dir, "pr_curves_all_models.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    # ---- Calibration: best model only (reliability diagram) ----
    y_score = test_scores[best_model_name]
    prob_true, prob_pred = calibration_curve(
        y_test, y_score, n_bins=20, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    blabel = leg.get(best_model_name, best_model_name)
    ax.plot(prob_pred, prob_true, "s-", label=blabel)
    ax.set_xlabel("Mean predicted PD (bin)")
    ax.set_ylabel("Fraction of positives (default rate in bin)")
    ax.set_title(f"Reliability / calibration — {blabel}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    p = os.path.join(out_dir, "calibration_best_model.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")
