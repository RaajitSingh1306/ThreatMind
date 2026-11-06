"""
Airflow DAG: CICIDS2017 ingestion and preprocessing pipeline.

Tasks:
  1. validate_csvs    — check archive/ CSVs exist and are readable
  2. run_transforms   — run spark_transforms.py (pandas/DuckDB mode locally)
  3. feature_engineer — apply feature engineering to Parquet files
  4. report_stats     — print label distribution and row counts
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.python import PythonOperator

from airflow import DAG

# Ensure project root is in sys.path for Airflow workers & IDE
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_ARGS = {
    "owner": "threatmind",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _validate_csvs(**context):
    import glob

    files = glob.glob("archive/*.csv")
    if not files:
        raise FileNotFoundError("No CSV files found in archive/")
    print(f"Found {len(files)} CSV files: {[f.split('/')[-1] for f in files]}")
    return len(files)


def _run_transforms(**context):
    from src.preprocessing.spark_transforms import run_transforms

    run_transforms(input_dir="archive", output_dir="data/processed")
    print("Transforms complete → data/processed/")


def _feature_engineer(**context):
    import pandas as pd

    from src.preprocessing.feature_engineering import add_flow_features

    for split in ["train", "val"]:
        path = f"data/processed/{split}.parquet"
        df = pd.read_parquet(path)
        df_eng = add_flow_features(df)
        df_eng.to_parquet(path, index=False)
        print(
            f"Feature engineering applied to {path}: {len(df_eng)} rows, {len(df_eng.columns)} cols"
        )


def _report_stats(**context):
    import pandas as pd

    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    print(f"Train: {len(train_df):,} rows")
    print(f"Val:   {len(val_df):,} rows")
    print("Label distribution (train):")
    print(train_df["label_str"].value_counts().to_string())


with DAG(
    dag_id="ingest_cicids",
    description="CICIDS2017 ingestion and preprocessing",
    default_args=DEFAULT_ARGS,
    schedule_interval="@weekly",
    catchup=False,
    tags=["threatmind", "data"],
) as dag:
    validate = PythonOperator(task_id="validate_csvs", python_callable=_validate_csvs)
    transform = PythonOperator(task_id="run_transforms", python_callable=_run_transforms)
    engineer = PythonOperator(task_id="feature_engineer", python_callable=_feature_engineer)
    report = PythonOperator(task_id="report_stats", python_callable=_report_stats)

    validate >> transform >> engineer >> report
