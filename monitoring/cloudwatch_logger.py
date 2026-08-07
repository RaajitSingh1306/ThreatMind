"""
CloudWatch logger + structlog integration for ThreatMind.

In local mode: logs to stdout via structlog.
In AWS mode: ships logs and custom metrics to CloudWatch.

Set AWS_REGION + S3_BUCKET env vars to enable CloudWatch mode.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

import structlog

log = structlog.get_logger()

# AWS CloudWatch available?
_CW_CLIENT = None
_CW_LOG_GROUP = "threatmind"
_CW_LOG_STREAM = f"api-{int(time.time())}"
_CW_NAMESPACE = "ThreatMind"


def _get_cw_client():
    """Lazy-init CloudWatch client. Returns None if boto3 not configured."""
    global _CW_CLIENT
    if _CW_CLIENT is not None:
        return _CW_CLIENT
    try:
        import boto3  # noqa: PLC0415

        region = os.getenv("AWS_REGION", "us-east-1")
        _CW_CLIENT = boto3.client("cloudwatch", region_name=region)
        log.info("CloudWatch client initialised", region=region)
        return _CW_CLIENT
    except Exception as e:
        log.debug("CloudWatch unavailable (local mode)", reason=str(e))
        return None


def put_metric(
    metric_name: str, value: float, unit: str = "Count", dimensions: dict | None = None
) -> None:
    """
    Publish a custom metric to CloudWatch.
    In local mode, just logs the metric.
    """
    dims = [{"Name": k, "Value": v} for k, v in (dimensions or {}).items()]
    log.info("metric", name=metric_name, value=value, unit=unit, dimensions=dims)

    cw = _get_cw_client()
    if cw is None:
        return  # local mode

    try:
        cw.put_metric_data(
            Namespace=_CW_NAMESPACE,
            MetricData=[
                {
                    "MetricName": metric_name,
                    "Value": value,
                    "Unit": unit,
                    "Dimensions": dims,
                }
            ],
        )
    except Exception as e:
        log.warning("CloudWatch put_metric failed", error=str(e))


class RequestMetricsCollector:
    """
    Collects per-request metrics and periodically flushes to CloudWatch.
    Thread-safe for asyncio (single-event-loop) usage.
    """

    def __init__(self, flush_interval_s: int = 60, window: int = 1000):
        self._latencies: deque[float] = deque(maxlen=window)
        self._predictions: deque[str] = deque(maxlen=window)
        self._errors: int = 0
        self._total: int = 0
        self._last_flush = time.time()
        self._flush_interval = flush_interval_s

    def record_request(
        self, latency_ms: float, prediction: str | None = None, error: bool = False
    ) -> None:
        self._total += 1
        self._latencies.append(latency_ms)
        if prediction:
            self._predictions.append(prediction)
        if error:
            self._errors += 1
        if time.time() - self._last_flush > self._flush_interval:
            self._flush()

    def _flush(self) -> None:
        if not self._latencies:
            return

        import numpy as np  # noqa: PLC0415

        lats = list(self._latencies)
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))
        error_rate = self._errors / max(self._total, 1)

        put_metric("Latency_p50", p50, unit="Milliseconds")
        put_metric("Latency_p95", p95, unit="Milliseconds")
        put_metric("Latency_p99", p99, unit="Milliseconds")
        put_metric("ErrorRate", error_rate, unit="None")
        put_metric("TotalRequests", self._total, unit="Count")

        log.info(
            "Metrics flushed", p50=p50, p95=p95, p99=p99, error_rate=error_rate, total=self._total
        )
        self._last_flush = time.time()

    def get_summary(self) -> dict[str, Any]:
        import numpy as np  # noqa: PLC0415

        if not self._latencies:
            return {}
        lats = list(self._latencies)
        return {
            "p50_ms": round(float(np.percentile(lats, 50)), 2),
            "p95_ms": round(float(np.percentile(lats, 95)), 2),
            "p99_ms": round(float(np.percentile(lats, 99)), 2),
            "error_rate": round(self._errors / max(self._total, 1), 4),
            "total_requests": self._total,
        }


# Singleton collector
METRICS = RequestMetricsCollector()
