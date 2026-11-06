"""
Pydantic schemas for ThreatMind FastAPI.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── /predict ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Network flow feature dictionary. Keys are CICIDS2017 feature names.",
        examples=[
            {
                "flow_duration": 0.002,
                "total_fwd_packets": 1.0,
                "total_bwd_packets": 0.0,
                "fwd_pkt_len_mean": 54.0,
                "syn_flag_count": 1.0,
            }
        ],
    )


class SHAPFeature(BaseModel):
    feature: str
    value: float
    shap: float


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Predicted attack category")
    label_id: int = Field(..., description="Numeric label ID")
    confidence: float = Field(..., description="Classifier confidence [0, 1]")
    anomaly_score: float = Field(..., description="Autoencoder anomaly score [0, 1]")
    is_anomaly: bool = Field(..., description="True if anomaly score exceeds threshold")
    shap_top_features: list[SHAPFeature] = Field(..., description="Top SHAP feature attributions")


# ── /agent/query ──────────────────────────────────────────────────────────────
class AgentQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Natural language threat intelligence query.",
        examples=["What CVEs are associated with Apache Log4j RCE and how severe are they?"],
    )


class AgentQueryResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list, description="Referenced CVE IDs")
    tool_calls: list[str] = Field(default_factory=list, description="Tools invoked by the agent")
    reasoning_trace: str = Field(default="", description="Agent reasoning steps")


# ── /health ───────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    autoencoder_loaded: bool
    rag_available: bool
    version: str = "0.1.0"


# ── /metrics ──────────────────────────────────────────────────────────────────
class MetricsResponse(BaseModel):
    total_requests: int
    predict_requests: int
    agent_requests: int
    avg_latency_ms: float | None
    error_rate: float | None
