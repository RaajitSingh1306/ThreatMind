"""
ThreatMind FastAPI application.

Endpoints:
  POST /predict      — ML classification + anomaly detection + SHAP
  POST /agent/query  — LangGraph ReAct agent
  GET  /health       — service health check
  GET  /metrics      — request metrics
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.serving.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    SHAPFeature,
)

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = structlog.get_logger()

# ── Global model state ─────────────────────────────────────────────────────────
_pipeline = None
_feature_names: list[str] = []
_detector = None
_label_names: dict[int, str] = {}

# Simple in-process metrics
_metrics: dict[str, Any] = {
    "total_requests": 0,
    "predict_requests": 0,
    "agent_requests": 0,
    "total_latency_ms": 0.0,
    "errors": 0,
}


# ── Startup / Shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _feature_names, _detector, _label_names

    model_dir = os.getenv("MODEL_DIR", "models")

    # Load sklearn pipeline
    pipeline_path = Path(model_dir) / "pipeline.joblib"
    if pipeline_path.exists():
        from src.ml.pipeline import LABEL_NAMES, load_pipeline  # noqa: PLC0415

        _pipeline = load_pipeline(str(pipeline_path))
        _label_names = LABEL_NAMES
        log.info("Pipeline loaded", path=str(pipeline_path))
    else:
        log.warning("Pipeline not found — /predict will fail", path=str(pipeline_path))

    # Load feature names
    feat_path = Path(model_dir) / "feature_names.json"
    if feat_path.exists():
        _feature_names = json.loads(feat_path.read_text())

    # Load autoencoder
    ae_path = Path(model_dir) / "autoencoder.pt"
    if ae_path.exists():
        from src.ml.autoencoder import AnomalyDetector  # noqa: PLC0415

        _detector = AnomalyDetector.load(str(ae_path))
        log.info("Autoencoder loaded", path=str(ae_path))
    else:
        log.warning("Autoencoder not found — anomaly detection disabled", path=str(ae_path))

    yield  # app running

    log.info("Shutting down ThreatMind API")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ThreatMind API",
    description="Cybersecurity Threat Intelligence — ML + LLM Agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: timing + metrics ───────────────────────────────────────────────
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start = time.perf_counter()
    _metrics["total_requests"] += 1
    try:
        response = await call_next(request)
        return response
    except Exception:
        _metrics["errors"] += 1
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _metrics["total_latency_ms"] += elapsed_ms


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    rag_ok = Path(os.getenv("CHROMA_PERSIST_DIR", "chroma_db")).exists()
    return HealthResponse(
        status="ok",
        pipeline_loaded=_pipeline is not None,
        autoencoder_loaded=_detector is not None,
        rag_available=rag_ok,
    )


@app.get("/metrics")
async def metrics():
    total = _metrics["total_requests"]
    avg_lat = _metrics["total_latency_ms"] / max(total, 1)
    return MetricsResponse(
        total_requests=total,
        predict_requests=_metrics["predict_requests"],
        agent_requests=_metrics["agent_requests"],
        avg_latency_ms=round(avg_lat, 2),
        error_rate=round(_metrics["errors"] / max(total, 1), 4),
    )


@app.post("/predict")
async def predict(request_body: PredictRequest):

    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    _metrics["predict_requests"] += 1

    # Build feature vector from request
    feat_dict = request_body.features
    if _feature_names:
        X = np.array([[feat_dict.get(f, 0.0) for f in _feature_names]], dtype=np.float32)
        feature_names = _feature_names
    else:
        feature_names = list(feat_dict.keys())
        X = np.array([list(feat_dict.values())], dtype=np.float32)

    try:
        y_pred = _pipeline.predict(X)[0]
        y_proba = _pipeline.predict_proba(X)[0]
        confidence = float(y_proba[y_pred])
        label = _label_names.get(int(y_pred), f"CLASS_{y_pred}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}") from e

    # Anomaly detection
    anomaly_score = 0.0
    is_anomaly = False
    if _detector is not None:
        try:
            scores = _detector.anomaly_score(X)
            anomaly_score = float(scores[0])
            flags, _ = _detector.predict_anomaly(X)
            is_anomaly = bool(flags[0])
        except Exception as e:
            log.warning("Autoencoder error", error=str(e))

    # SHAP
    shap_features: list[SHAPFeature] = []
    try:
        from src.ml.pipeline import get_shap_top_features  # noqa: PLC0415

        shap_data = get_shap_top_features(_pipeline, X, feature_names, top_n=10)
        shap_features = [SHAPFeature(**s) for s in shap_data]
    except Exception as e:
        log.warning("SHAP error", error=str(e))

    log.info("Prediction", label=label, confidence=round(confidence, 3), anomaly_score=round(anomaly_score, 3))

    return PredictResponse(
        prediction=label,
        label_id=int(y_pred),
        confidence=round(confidence, 4),
        anomaly_score=round(anomaly_score, 4),
        is_anomaly=is_anomaly,
        shap_top_features=shap_features,
    )


@app.post("/agent/query")
async def agent_query(request_body: AgentQueryRequest):
    from src.agent.graph import run_agent  # noqa: PLC0415

    _metrics["agent_requests"] += 1

    try:
        result = run_agent(request_body.query)
    except Exception as e:
        log.error("Agent error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {e}") from e

    return AgentQueryResponse(**result)


if __name__ == "__main__":
    uvicorn.run(
        "src.serving.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
