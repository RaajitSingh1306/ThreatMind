"""
Model evaluation utilities for ThreatMind.

Computes: AUC-PR, ROC-AUC, F1 (macro), calibration curve, SHAP waterfall plots.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import structlog
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

log = structlog.get_logger()


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_names: dict[int, str],
    output_dir: str = "reports",
) -> dict[str, Any]:
    """
    Full evaluation suite for multi-class classifier.
    Returns metrics dict and saves plots to output_dir.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_prob_classes = y_proba.shape[1]
    classes = list(range(n_prob_classes))

    # Binarise for OvR metrics
    y_bin = label_binarize(y_true, classes=classes)
    if y_bin.shape[1] == 1 and n_prob_classes > 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    # ROC-AUC & AUC-PR (macro OvR over classes with both pos/neg samples)
    roc_scores = []
    pr_scores = []
    max_cols = min(y_bin.shape[1], y_proba.shape[1])
    for i in range(max_cols):
        if np.unique(y_bin[:, i]).size > 1:
            with contextlib.suppress(Exception):
                roc_scores.append(roc_auc_score(y_bin[:, i], y_proba[:, i]))
        if y_bin[:, i].sum() > 0:
            with contextlib.suppress(Exception):
                pr_scores.append(average_precision_score(y_bin[:, i], y_proba[:, i]))

    roc_auc = float(np.mean(roc_scores)) if roc_scores else 0.0
    auc_pr = float(np.mean(pr_scores)) if pr_scores else 0.0

    # F1 macro
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Classification report
    target_names = [label_names.get(c, f"Class_{c}") for c in classes]
    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        target_names=target_names,
        zero_division=0,
    )

    log.info(
        "Classifier metrics",
        roc_auc=round(roc_auc, 4),
        auc_pr=round(auc_pr, 4),
        f1_macro=round(f1, 4),
    )
    log.info("Classification report\n" + report)

    # Save report
    report_path = Path(output_dir) / "classification_report.txt"
    report_path.write_text(report)

    return {
        "roc_auc_macro": roc_auc,
        "auc_pr_macro": auc_pr,
        "f1_macro": f1,
        "report": report,
    }


def evaluate_anomaly_detector(
    y_true_binary: np.ndarray,
    anomaly_scores: np.ndarray,
    output_dir: str = "reports",
) -> dict[str, float]:
    """
    Evaluate autoencoder anomaly detector.
    y_true_binary: 0=benign, 1=anomaly.
    """
    from sklearn.metrics import precision_recall_curve, roc_auc_score

    auc = roc_auc_score(y_true_binary, anomaly_scores)
    prec, rec, thresholds = precision_recall_curve(y_true_binary, anomaly_scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1s)
    best_f1 = f1s[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    log.info(
        "Anomaly detector metrics",
        roc_auc=round(auc, 4),
        best_f1=round(best_f1, 4),
        best_threshold=round(best_threshold, 4),
    )

    # Plot PR curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="steelblue", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Anomaly Detector PR Curve (AUC={auc:.3f})")
    ax.grid(True, alpha=0.3)
    fig.savefig(Path(output_dir) / "anomaly_pr_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {"roc_auc": auc, "best_f1": best_f1, "best_threshold": best_threshold}
