"""
MLflow experiment entry point for ThreatMind.

Usage:
    python src/ml/train.py --experiment-name threatmind-v1

Trains:
  1. sklearn Pipeline (XGBoost + LightGBM) on full CICIDS2017 features
  2. PyTorch Autoencoder on BENIGN-only samples

Logs all metrics, params, and artifacts to MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ml.autoencoder import AnomalyDetector  # noqa: E402
from src.ml.evaluation import evaluate_anomaly_detector, evaluate_classifier  # noqa: E402
from src.ml.pipeline import LABEL_NAMES, build_pipeline, save_pipeline  # noqa: E402
from src.preprocessing.feature_engineering import (  # noqa: E402
    ENGINEERED_FEATURES,
    add_flow_features,
)
from src.preprocessing.spark_transforms import FEATURE_COLS  # noqa: E402


def load_data(processed_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(Path(processed_dir) / "train.parquet")
    val = pd.read_parquet(Path(processed_dir) / "val.parquet")
    log.info("Data loaded", train_rows=len(train), val_rows=len(val))
    return train, val


def get_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = add_flow_features(df)
    base_features = [c for c in FEATURE_COLS if c in df.columns]
    eng_features = [c for c in ENGINEERED_FEATURES if c in df.columns]
    all_features = base_features + eng_features
    X = df[all_features].values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y, all_features


def train(
    processed_dir: str = "data/processed",
    model_dir: str = "models",
    experiment_name: str = "threatmind-v1",
    autoencoder_epochs: int = 15,
) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))
    mlflow.set_experiment(experiment_name)

    train_df, val_df = load_data(processed_dir)
    X_train, y_train, feature_names = get_feature_matrix(train_df)
    X_val, y_val, _ = get_feature_matrix(val_df)

    n_classes = len(np.unique(y_train))
    log.info(
        "Training",
        n_train=len(X_train),
        n_val=len(X_val),
        n_classes=n_classes,
        n_features=len(feature_names),
    )

    # ── 1. XGBoost + LGB Pipeline ─────────────────────────────────────────────
    with mlflow.start_run(run_name="xgb_lgb_pipeline"):
        mlflow.log_param("n_classes", n_classes)
        mlflow.log_param("n_features", len(feature_names))

        pipeline = build_pipeline(n_classes=n_classes)
        log.info("Fitting sklearn pipeline...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)

        metrics = evaluate_classifier(y_val, y_pred, y_proba, LABEL_NAMES, output_dir="reports")
        mlflow.log_metrics(
            {
                "roc_auc_macro": float(metrics["roc_auc_macro"]),
                "auc_pr_macro": float(metrics["auc_pr_macro"]),
                "f1_macro": float(metrics["f1_macro"]),
            }
        )
        mlflow.log_artifact("reports/classification_report.txt")

        pipeline_path = Path(model_dir) / "pipeline.joblib"
        save_pipeline(pipeline, str(pipeline_path))
        mlflow.sklearn.log_model(
            pipeline, "pipeline", registered_model_name="ThreatMind-Classifier"
        )

        # Save feature names for inference
        feat_path = Path(model_dir) / "feature_names.json"
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        feat_path.write_text(json.dumps(feature_names))
        mlflow.log_artifact(str(feat_path))

    # ── 2. Autoencoder ────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="autoencoder"):
        # Train on BENIGN only
        benign_mask = y_train == 0
        X_benign = X_train[benign_mask]
        log.info("Training autoencoder on benign samples", n=len(X_benign))

        detector = AnomalyDetector(input_dim=X_benign.shape[1])
        mlflow.log_param("latent_dim", 64)
        mlflow.log_param("epochs", autoencoder_epochs)
        mlflow.log_param("threshold_percentile", 95)

        losses = detector.fit(X_benign, epochs=autoencoder_epochs)
        for i, loss in enumerate(losses):
            mlflow.log_metric("train_loss", loss, step=i)

        # Evaluate on val: anomaly = non-BENIGN
        y_binary = (y_val != 0).astype(int)
        scores = detector.anomaly_score(X_val)
        ae_metrics = evaluate_anomaly_detector(y_binary, scores, output_dir="reports")

        mlflow.log_metrics({"roc_auc": ae_metrics["roc_auc"], "best_f1": ae_metrics["best_f1"]})
        mlflow.log_artifact("reports/anomaly_pr_curve.png")

        ae_path = Path(model_dir) / "autoencoder.pt"
        detector.save(str(ae_path))
        mlflow.log_artifact(str(ae_path))

    log.info("Training complete. All artifacts saved.", model_dir=model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="ThreatMind model training")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--experiment-name", default="threatmind-v1")
    parser.add_argument("--ae-epochs", type=int, default=15)
    args = parser.parse_args()

    train(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        experiment_name=args.experiment_name,
        autoencoder_epochs=args.ae_epochs,
    )


if __name__ == "__main__":
    main()
