"""
Airflow DAG: ThreatMind model training pipeline.
Triggered after ingest_cicids completes.

Tasks:
  1. train_classifier  — fit XGBoost/LGB pipeline + log to MLflow
  2. train_autoencoder — train PyTorch autoencoder on BENIGN flows
  3. validate_models   — sanity-check saved model artifacts
  4. notify            — print summary
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

DEFAULT_ARGS = {
    "owner": "threatmind",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _train_classifier(**context):
    from src.ml.train import load_data, get_feature_matrix  # noqa: PLC0415
    from src.ml.pipeline import build_pipeline, save_pipeline, LABEL_NAMES  # noqa: PLC0415
    from src.ml.evaluation import evaluate_classifier  # noqa: PLC0415
    import mlflow
    import os
    import json
    from pathlib import Path
    import numpy as np

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "threatmind-v1"))

    train_df, val_df = load_data("data/processed")
    X_train, y_train, feature_names = get_feature_matrix(train_df)
    X_val, y_val, _ = get_feature_matrix(val_df)
    n_classes = len(np.unique(y_train))

    with mlflow.start_run(run_name="xgb_lgb_pipeline_airflow"):
        pipeline = build_pipeline(n_classes=n_classes)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)
        metrics = evaluate_classifier(y_val, y_pred, y_proba, LABEL_NAMES, output_dir="reports")
        mlflow.log_metrics(metrics)
        save_pipeline(pipeline, "models/pipeline.joblib")
        Path("models/feature_names.json").write_text(json.dumps(feature_names))

    print(f"Classifier trained. F1={metrics['f1_macro']:.4f}")


def _train_autoencoder(**context):
    from src.ml.train import load_data, get_feature_matrix  # noqa: PLC0415
    from src.ml.autoencoder import AnomalyDetector  # noqa: PLC0415
    import numpy as np

    train_df, _ = load_data("data/processed")
    X_train, y_train, _ = get_feature_matrix(train_df)
    X_benign = X_train[y_train == 0]

    detector = AnomalyDetector(input_dim=X_benign.shape[1])
    detector.fit(X_benign, epochs=15)
    detector.save("models/autoencoder.pt")
    print(f"Autoencoder trained on {len(X_benign):,} benign samples. Threshold: {detector.threshold:.6f}")


def _validate_models(**context):
    from pathlib import Path

    required = ["models/pipeline.joblib", "models/autoencoder.pt", "models/feature_names.json"]
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing model artifact: {path}")
        size_kb = Path(path).stat().st_size / 1024
        print(f"✓ {path} ({size_kb:.1f} KB)")
    print("All model artifacts validated.")


def _notify(**context):
    print("=== ThreatMind Training Pipeline Complete ===")
    print("Models saved to: models/")
    print("Metrics logged to MLflow.")
    print("Run: uvicorn src.serving.api:app --reload --port 8000")


with DAG(
    dag_id="train_pipeline",
    description="ThreatMind model training pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,  # triggered by ingest_cicids
    catchup=False,
    tags=["threatmind", "training"],
) as dag:
    wait_for_ingest = ExternalTaskSensor(
        task_id="wait_for_ingest",
        external_dag_id="ingest_cicids",
        external_task_id="report_stats",
        timeout=3600,
        mode="poke",
        poke_interval=60,
    )
    train_clf = PythonOperator(task_id="train_classifier", python_callable=_train_classifier)
    train_ae = PythonOperator(task_id="train_autoencoder", python_callable=_train_autoencoder)
    validate = PythonOperator(task_id="validate_models", python_callable=_validate_models)
    notify = PythonOperator(task_id="notify", python_callable=_notify)

    wait_for_ingest >> [train_clf, train_ae] >> validate >> notify
