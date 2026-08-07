"""
Integration tests for the FastAPI application.
Uses httpx.AsyncClient with the FastAPI TestClient.
Models are NOT required — endpoints degrade gracefully.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.serving.api import app

    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "pipeline_loaded" in data
        assert "autoencoder_loaded" in data
        assert "rag_available" in data
        assert data["status"] == "ok"


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_schema(self, client):
        resp = client.get("/metrics")
        data = resp.json()
        assert "total_requests" in data
        assert "predict_requests" in data
        assert "agent_requests" in data


class TestPredictEndpoint:
    def test_predict_no_model_returns_503(self, client):
        """Without trained models, /predict should return 503."""
        payload = {
            "features": {
                "flow_duration": 0.002,
                "total_fwd_packets": 1.0,
                "total_bwd_packets": 0.0,
            }
        }
        resp = client.post("/predict", json=payload)
        # 503 if model not loaded, 200 if it is
        assert resp.status_code in (200, 503)

    def test_predict_invalid_payload(self, client):
        """Missing 'features' field should return 422."""
        resp = client.post("/predict", json={"not_features": {}})
        assert resp.status_code == 422

    def test_predict_empty_features(self, client):
        """Empty features dict: 503 (no model) or 200 (model returns default)."""
        resp = client.post("/predict", json={"features": {}})
        assert resp.status_code in (200, 503)


class TestAgentQueryEndpoint:
    def test_agent_query_short_query_rejected(self, client):
        """Very short query should be rejected by Pydantic validation."""
        resp = client.post("/agent/query", json={"query": "hi"})
        assert resp.status_code == 422

    def test_agent_query_missing_llm_key(self, client):
        """Agent without LLM key should return 500 error (gracefully)."""
        resp = client.post(
            "/agent/query",
            json={"query": "What is Log4Shell and what CVEs are associated with it?"},
        )
        assert resp.status_code in (200, 500)
