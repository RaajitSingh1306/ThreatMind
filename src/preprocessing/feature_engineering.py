"""
Feature engineering for ThreatMind network flow data.

Derives higher-order flow-level features from base CICIDS2017 columns.
Can be applied to both train and inference DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to a cleaned CICIDS DataFrame.
    All operations are vectorised for speed.
    """
    df = df.copy()

    # Packet asymmetry ratio — high for SYN floods (all fwd, no bwd)
    total_pkts = df["total_fwd_packets"] + df["total_bwd_packets"]
    df["pkt_asymmetry"] = (df["total_fwd_packets"] - df["total_bwd_packets"]) / total_pkts.clip(lower=1)

    # Byte asymmetry
    total_bytes = df["total_len_fwd_packets"] + df["total_len_bwd_packets"]
    df["byte_asymmetry"] = (df["total_len_fwd_packets"] - df["total_len_bwd_packets"]) / total_bytes.clip(lower=1)

    # Mean packet size across both directions
    df["overall_avg_pkt_size"] = total_bytes / total_pkts.clip(lower=1)

    # Flow duration in seconds (raw is microseconds)
    df["flow_duration_s"] = df["flow_duration"] / 1e6

    # Packets per second (guard against zero-duration)
    df["total_pkts_per_s"] = total_pkts / df["flow_duration_s"].clip(lower=1e-9)

    # Flag density — ratio of flag-set packets to total
    flag_cols = ["syn_flag_count", "fin_flag_count", "rst_flag_count", "ack_flag_count"]
    for col in flag_cols:
        if col in df.columns:
            df[f"{col}_ratio"] = df[col] / total_pkts.clip(lower=1)

    # IAT coefficient of variation (captures burstiness)
    if "flow_iat_mean" in df.columns and "flow_iat_std" in df.columns:
        df["iat_cv"] = df["flow_iat_std"] / df["flow_iat_mean"].clip(lower=1e-9)

    # Header overhead ratio
    if "fwd_header_len" in df.columns and "total_len_fwd_packets" in df.columns:
        df["fwd_header_ratio"] = df["fwd_header_len"] / df["total_len_fwd_packets"].clip(lower=1)

    # Replace any newly introduced inf
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.fillna(0.0, inplace=True)

    return df


ENGINEERED_FEATURES: list[str] = [
    "pkt_asymmetry",
    "byte_asymmetry",
    "overall_avg_pkt_size",
    "flow_duration_s",
    "total_pkts_per_s",
    "syn_flag_count_ratio",
    "fin_flag_count_ratio",
    "rst_flag_count_ratio",
    "ack_flag_count_ratio",
    "iat_cv",
    "fwd_header_ratio",
]
