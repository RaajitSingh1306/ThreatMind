"""
Unit tests for agent tools (with mocked HTTP calls).
"""

from unittest.mock import MagicMock, patch

import pytest


class TestMLInferenceTool:
    @patch("src.agent.tools.httpx.post")
    def test_ml_inference_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "prediction": "DoS Hulk",
                "confidence": 0.95,
                "anomaly_score": 0.7,
                "is_anomaly": True,
                "shap_top_features": [],
            },
        )
        mock_post.return_value.raise_for_status = MagicMock()

        from src.agent.tools import ml_inference

        result = ml_inference.invoke({"features": {"flow_duration": 0.002, "total_fwd_packets": 1.0}})
        assert result["prediction"] == "DoS Hulk"
        assert result["confidence"] == 0.95

    @patch("src.agent.tools.httpx.post")
    def test_ml_inference_error_returns_dict(self, mock_post):
        mock_post.side_effect = Exception("connection refused")
        from src.agent.tools import ml_inference

        result = ml_inference.invoke({"features": {"flow_duration": 0.002}})
        assert "error" in result


class TestNVDLookupTool:
    @patch("src.agent.tools.requests.get")
    def test_nvd_lookup_success(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "totalResults": 1,
                "vulnerabilities": [
                    {
                        "cve": {
                            "id": "CVE-2021-44228",
                            "descriptions": [{"lang": "en", "value": "Log4Shell critical RCE vulnerability in Apache Log4j."}],
                            "metrics": {},
                            "published": "2021-12-10T00:00:00.000",
                        }
                    }
                ],
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        from src.agent.tools import nvd_lookup

        results = nvd_lookup.invoke({"keyword": "Log4j", "max_results": 3})
        assert len(results) >= 1
        assert results[0]["cve_id"] == "CVE-2021-44228"

    @patch("src.agent.tools.requests.get")
    def test_nvd_lookup_error(self, mock_get):
        mock_get.side_effect = Exception("network error")
        from src.agent.tools import nvd_lookup

        results = nvd_lookup.invoke({"keyword": "test"})
        assert len(results) == 1
        assert "error" in results[0]


class TestDriftDetector:
    def test_no_drift(self):
        import numpy as np

        from monitoring.drift_detector import DriftDetector

        baseline_proba = np.random.rand(1000, 10)
        baseline_proba = baseline_proba / baseline_proba.sum(axis=1, keepdims=True)
        detector = DriftDetector.from_training_data(baseline_proba)

        # Similar distribution — should not drift
        current = np.random.rand(100, 10)
        current = current / current.sum(axis=1, keepdims=True)
        result = detector.check(current)
        assert "kl_divergence" in result

    def test_drift_detected(self):
        import numpy as np

        from monitoring.drift_detector import DriftDetector

        # Very uniform baseline
        baseline_proba = np.ones((500, 5)) / 5
        detector = DriftDetector.from_training_data(baseline_proba, kl_threshold=0.001)

        # Very skewed current — should trigger drift
        skewed = np.zeros((100, 5))
        skewed[:, 0] = 1.0
        result = detector.check(skewed)
        assert result["is_drifting"] is True
