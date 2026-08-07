"""
PySpark-style preprocessing pipeline for CICIDS2017 dataset.

Runs either via PySpark (production) or pandas/DuckDB (local dev).
Usage:
    python src/preprocessing/spark_transforms.py --input archive/ --output data/processed/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

# ─── Column renames (strip leading spaces, normalise to snake_case) ───────────
LABEL_COL = " Label"

RENAME_MAP = {
    " Destination Port": "destination_port",
    " Flow Duration": "flow_duration",
    " Total Fwd Packets": "total_fwd_packets",
    " Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_len_fwd_packets",
    " Total Length of Bwd Packets": "total_len_bwd_packets",
    " Fwd Packet Length Max": "fwd_pkt_len_max",
    " Fwd Packet Length Min": "fwd_pkt_len_min",
    " Fwd Packet Length Mean": "fwd_pkt_len_mean",
    " Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    " Bwd Packet Length Min": "bwd_pkt_len_min",
    " Bwd Packet Length Mean": "bwd_pkt_len_mean",
    " Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_bytes_per_s",
    " Flow Packets/s": "flow_pkts_per_s",
    " Flow IAT Mean": "flow_iat_mean",
    " Flow IAT Std": "flow_iat_std",
    " Flow IAT Max": "flow_iat_max",
    " Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_total",
    " Fwd IAT Mean": "fwd_iat_mean",
    " Fwd IAT Std": "fwd_iat_std",
    " Fwd IAT Max": "fwd_iat_max",
    " Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_total",
    " Bwd IAT Mean": "bwd_iat_mean",
    " Bwd IAT Std": "bwd_iat_std",
    " Bwd IAT Max": "bwd_iat_max",
    " Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    " Bwd PSH Flags": "bwd_psh_flags",
    " Fwd URG Flags": "fwd_urg_flags",
    " Bwd URG Flags": "bwd_urg_flags",
    " Fwd Header Length": "fwd_header_len",
    " Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_pkts_per_s",
    " Bwd Packets/s": "bwd_pkts_per_s",
    " Min Packet Length": "min_pkt_len",
    " Max Packet Length": "max_pkt_len",
    " Packet Length Mean": "pkt_len_mean",
    " Packet Length Std": "pkt_len_std",
    " Packet Length Variance": "pkt_len_variance",
    "FIN Flag Count": "fin_flag_count",
    " SYN Flag Count": "syn_flag_count",
    " RST Flag Count": "rst_flag_count",
    " PSH Flag Count": "psh_flag_count",
    " ACK Flag Count": "ack_flag_count",
    " URG Flag Count": "urg_flag_count",
    " CWE Flag Count": "cwe_flag_count",
    " ECE Flag Count": "ece_flag_count",
    " Down/Up Ratio": "down_up_ratio",
    " Average Packet Size": "avg_pkt_size",
    " Avg Fwd Segment Size": "avg_fwd_segment_size",
    " Avg Bwd Segment Size": "avg_bwd_segment_size",
    " Fwd Header Length.1": "fwd_header_len2",
    "Fwd Avg Bytes/Bulk": "fwd_avg_bytes_bulk",
    " Fwd Avg Packets/Bulk": "fwd_avg_pkts_bulk",
    " Fwd Avg Bulk Rate": "fwd_avg_bulk_rate",
    " Bwd Avg Bytes/Bulk": "bwd_avg_bytes_bulk",
    " Bwd Avg Packets/Bulk": "bwd_avg_pkts_bulk",
    "Bwd Avg Bulk Rate": "bwd_avg_bulk_rate",
    "Subflow Fwd Packets": "subflow_fwd_pkts",
    " Subflow Fwd Bytes": "subflow_fwd_bytes",
    " Subflow Bwd Packets": "subflow_bwd_pkts",
    " Subflow Bwd Bytes": "subflow_bwd_bytes",
    "Init_Win_bytes_forward": "init_win_bytes_fwd",
    " Init_Win_bytes_backward": "init_win_bytes_bwd",
    " act_data_pkt_fwd": "act_data_pkt_fwd",
    " min_seg_size_forward": "min_seg_size_fwd",
    "Active Mean": "active_mean",
    " Active Std": "active_std",
    " Active Max": "active_max",
    " Active Min": "active_min",
    "Idle Mean": "idle_mean",
    " Idle Std": "idle_std",
    " Idle Max": "idle_max",
    " Idle Min": "idle_min",
    " Label": "label",
}

# Attack label → integer mapping
LABEL_MAP = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "PortScan": 2,
    "DDoS": 3,
    "DoS GoldenEye": 4,
    "FTP-Patator": 5,
    "SSH-Patator": 6,
    "DoS slowloris": 7,
    "DoS Slowhttptest": 8,
    "Bot": 9,
    "Web Attack \x96 Brute Force": 10,
    "Web Attack \x96 XSS": 11,
    "Web Attack \x96 Sql Injection": 12,
    "Infiltration": 13,
    "Heartbleed": 14,
}

FEATURE_COLS: list[str] = [v for k, v in RENAME_MAP.items() if k != " Label"]


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a single CICIDS CSV with encoding fallback."""
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, fix dtypes, drop NaN/Inf rows, encode label."""
    # Rename
    df = df.rename(columns=RENAME_MAP)

    # Encode label
    df["label"] = df["label"].str.strip()
    df["label_str"] = df["label"]
    df["label"] = df["label"].map(LABEL_MAP).fillna(-1).astype(int)
    df = df[df["label"] >= 0].copy()  # drop unknown labels

    # Numeric cast
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN in features
    df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns], inplace=True)

    return df


def run_transforms(input_dir: str, output_dir: str) -> None:
    """
    Load all CSVs from input_dir, clean & combine, write Parquet to output_dir.
    Uses DuckDB for fast column selection and label stats.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        log.error("No CSV files found", input_dir=str(input_path))
        sys.exit(1)

    log.info("Found CSV files", count=len(csv_files), files=[f.name for f in csv_files])

    frames: list[pd.DataFrame] = []
    for f in csv_files:
        log.info("Loading", file=f.name, size_mb=round(f.stat().st_size / 1e6, 1))
        df = _load_csv(f)
        df = _clean_df(df)
        frames.append(df)
        log.info("Cleaned", file=f.name, rows=len(df))

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined dataset", total_rows=len(combined), label_dist=combined["label"].value_counts().to_dict())

    # Write partitioned Parquet
    train_frac = 0.8
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(combined) * train_frac)
    train_df = combined.iloc[:split_idx]
    val_df = combined.iloc[split_idx:]

    train_out = output_path / "train.parquet"
    val_out = output_path / "val.parquet"
    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)

    log.info("Wrote Parquet files", train=str(train_out), val=str(val_out), train_rows=len(train_df), val_rows=len(val_df))

    # DuckDB quick stats
    con = duckdb.connect()
    stats = con.execute(f"SELECT label_str, COUNT(*) as n FROM read_parquet('{train_out}') GROUP BY label_str ORDER BY n DESC").fetchdf()
    log.info("Label distribution (train)", stats=stats.to_dict("records"))


def main() -> None:
    parser = argparse.ArgumentParser(description="ThreatMind — CICIDS2017 preprocessing")
    parser.add_argument("--input", default="archive", help="Directory containing raw CSVs")
    parser.add_argument("--output", default="data/processed", help="Output directory for Parquet files")
    args = parser.parse_args()
    run_transforms(args.input, args.output)


if __name__ == "__main__":
    main()
