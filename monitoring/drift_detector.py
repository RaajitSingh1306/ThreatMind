"""
Prediction distribution drift detector for ThreatMind.

Computes KL divergence between the current prediction score distribution
and the training baseline. Triggers alert if KL > threshold.

Usage:
    detector = DriftDetector.from_training_data(y_proba_train)
    alert = detector.check(y_proba_current_window)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger()

# KL divergence threshold for alert
DEFAULT_KL_THRESHOLD = 0.1
EPSILON = 1e-10  # Laplace smoothing to avoid log(0)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Symmetric KL divergence: (KL(P||Q) + KL(Q||P)) / 2
    Both p and q should be 1D probability distributions (summing to 1).
    """
    p = np.clip(p, EPSILON, None)
    q = np.clip(q, EPSILON, None)
    p = p / p.sum()
    q = q / q.sum()
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    return float((kl_pq + kl_qp) / 2)


def _score_histogram(y_proba: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """Convert probability scores (max class prob) to a histogram."""
    max_proba = y_proba.max(axis=1) if y_proba.ndim == 2 else y_proba
    hist, _ = np.histogram(max_proba, bins=n_bins, range=(0.0, 1.0), density=False)
    return hist.astype(float) + EPSILON  # smoothing


class DriftDetector:
    """
    Detects covariate / label drift using KL divergence on prediction score distribution.
    """

    def __init__(self, baseline_hist: np.ndarray, n_bins: int = 50, kl_threshold: float = DEFAULT_KL_THRESHOLD):
        self.baseline_hist = baseline_hist
        self.n_bins = n_bins
        self.kl_threshold = kl_threshold
        self._history: list[dict] = []

    @classmethod
    def from_training_data(
        cls,
        y_proba_train: np.ndarray,
        n_bins: int = 50,
        kl_threshold: float = DEFAULT_KL_THRESHOLD,
    ) -> DriftDetector:
        hist = _score_histogram(y_proba_train, n_bins)
        log.info("Drift detector baseline set", n_samples=len(y_proba_train), n_bins=n_bins)
        return cls(hist, n_bins, kl_threshold)

    @classmethod
    def load(cls, path: str) -> DriftDetector:
        data = json.loads(Path(path).read_text())
        return cls(
            baseline_hist=np.array(data["baseline_hist"]),
            n_bins=data["n_bins"],
            kl_threshold=data["kl_threshold"],
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "baseline_hist": self.baseline_hist.tolist(),
            "n_bins": self.n_bins,
            "kl_threshold": self.kl_threshold,
        }))
        log.info("Drift detector saved", path=path)

    def check(self, y_proba_window: np.ndarray, window_label: str = "") -> dict:
        """
        Compute KL divergence between current window and baseline.
        Returns dict with kl_divergence, is_drifting, and alert.
        """
        if len(y_proba_window) < 10:
            return {"kl_divergence": 0.0, "is_drifting": False, "reason": "insufficient data"}

        current_hist = _score_histogram(y_proba_window, self.n_bins)
        kl = kl_divergence(self.baseline_hist, current_hist)
        is_drifting = kl > self.kl_threshold

        result = {
            "kl_divergence": round(kl, 6),
            "is_drifting": is_drifting,
            "threshold": self.kl_threshold,
            "window_size": len(y_proba_window),
            "window_label": window_label,
        }

        self._history.append(result)

        if is_drifting:
            log.warning(
                "DRIFT DETECTED",
                kl_divergence=round(kl, 6),
                threshold=self.kl_threshold,
                window=window_label,
            )
            # Emit CloudWatch metric
            try:
                from monitoring.cloudwatch_logger import put_metric  # noqa: PLC0415

                put_metric("KL_Divergence", kl, unit="None")
                put_metric("DriftAlert", 1.0, unit="Count")
            except Exception:
                pass
        else:
            log.info("Drift check OK", kl_divergence=round(kl, 6), window=window_label)

        return result

    def get_history(self) -> list[dict]:
        return self._history.copy()
