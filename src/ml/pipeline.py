"""
sklearn Pipeline with XGBoost + LightGBM ensemble for CICIDS2017 classification.

The pipeline handles:
- Median imputation of missing values
- Standard scaling
- XGBoost multi-class classifier (primary)
- LightGBM multi-class classifier (ensemble member)
- Soft-voting ensemble
- SHAP value computation on the XGBoost model
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import structlog
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

log = structlog.get_logger()

try:
    from lightgbm import LGBMClassifier

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    log.warning("LightGBM not installed — using XGBoost only")


LABEL_NAMES = {
    0: "BENIGN",
    1: "DoS Hulk",
    2: "PortScan",
    3: "DDoS",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
    6: "SSH-Patator",
    7: "DoS slowloris",
    8: "DoS Slowhttptest",
    9: "Bot",
    10: "Web Attack - Brute Force",
    11: "Web Attack - XSS",
    12: "Web Attack - SQL Injection",
    13: "Infiltration",
    14: "Heartbleed",
}


def build_pipeline(n_classes: int = 15, n_jobs: int = -1) -> Pipeline:
    """
    Build the full sklearn Pipeline:
      Imputer → Scaler → VotingClassifier(XGBoost [+ LightGBM])
    """
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=42,
        num_class=n_classes,
        objective="multi:softprob",
    )

    if _HAS_LGB:
        lgb = LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=n_jobs,
            random_state=42,
            verbose=-1,
        )
        clf = VotingClassifier(
            estimators=[("xgb", xgb), ("lgb", lgb)],
            voting="soft",
            n_jobs=1,  # inner models already parallelise
        )
    else:
        clf = xgb

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ]
    )
    return pipeline


def get_shap_top_features(
    pipeline: Pipeline,
    X_sample: np.ndarray,
    feature_names: list[str],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """
    Compute SHAP values for a single sample using the XGBoost sub-model.
    Returns list of {feature, value, shap} sorted by |shap| descending.
    """
    # Extract XGBoost from VotingClassifier or directly
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "estimators_") and hasattr(clf, "estimators"):
        # estimators: list of (name, estimator) from init
        # estimators_: list of fitted estimator objects (same order)
        name_to_fitted = {name: fitted for (name, _), fitted in zip(clf.estimators, clf.estimators_)}
        xgb_model = name_to_fitted["xgb"]
    else:
        xgb_model = clf

    # Transform through imputer + scaler only
    X_transformed = pipeline[:-1].transform(X_sample)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)

    # Newer SHAP (>=0.44) returns ndarray of shape:
    #   (n_samples, n_features) for binary
    #   (n_samples, n_features, n_classes) for multi-class
    # Older SHAP returns a list of per-class arrays.
    sv = np.array(shap_values) if not isinstance(shap_values, np.ndarray) else shap_values
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes) → mean |shap| across classes
        mean_shap_1d = np.mean(np.abs(sv[0]), axis=-1)  # shape: (n_features,)
    elif sv.ndim == 2:
        mean_shap_1d = np.abs(sv[0])  # shape: (n_features,)
    else:
        # list of (n_samples, n_features) arrays — old SHAP API
        sv_stack = np.stack([np.abs(arr[0]) for arr in shap_values], axis=-1)  # (n_features, n_classes)
        mean_shap_1d = np.mean(sv_stack, axis=-1)

    assert mean_shap_1d.ndim == 1, f"Expected 1D mean_shap, got shape {mean_shap_1d.shape}"
    sorted_indices = list(np.argsort(mean_shap_1d)[::-1][:top_n])

    return [
        {
            "feature": feature_names[int(idx)],
            "value": float(X_sample[0, int(idx)]),
            "shap": round(float(mean_shap_1d[int(idx)]), 4),
        }
        for idx in sorted_indices
    ]


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    log.info("Saved pipeline", path=path)


def load_pipeline(path: str) -> Pipeline:
    pipeline = joblib.load(path)
    log.info("Loaded pipeline", path=path)
    return pipeline
