"""
LangGraph agent tools for ThreatMind.

Three tools:
  1. ml_inference   — calls FastAPI /predict
  2. rag_retrieval  — queries ChromaDB via ThreatRAG
  3. nvd_lookup     — queries NVD REST API for CVE metadata
"""

from __future__ import annotations

import os
from typing import Annotated, Any

import httpx
import requests
import structlog
from langchain_core.tools import tool

log = structlog.get_logger()

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"


@tool
def ml_inference(features: dict[str, float]) -> dict[str, Any]:
    """
    Run ML inference on a network flow feature dict.
    Returns prediction, confidence, anomaly_score, and top SHAP features.

    Args:
        features: Dictionary of network flow feature names to float values.
                  Required keys include: flow_duration, total_fwd_packets, total_bwd_packets, etc.
    """
    try:
        resp = httpx.post(f"{API_BASE}/predict", json={"features": features}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error("ml_inference error", error=str(e))
        return {"error": str(e)}


@tool
def rag_retrieval(query: str) -> dict[str, Any]:
    """
    Query the CVE/threat intelligence RAG pipeline.
    Returns a synthesised answer with source CVE IDs.

    Args:
        query: Natural language threat intelligence question.
               Example: "What CVEs are associated with Apache Log4j RCE?"
    """
    try:
        from src.rag.retriever import ThreatRAG  # noqa: PLC0415

        rag = ThreatRAG()
        return rag.answer(query)
    except Exception as e:
        log.error("rag_retrieval error", error=str(e))
        return {"error": str(e), "answer": f"RAG unavailable: {e}"}


@tool
def nvd_lookup(keyword: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Look up CVEs from the National Vulnerability Database by keyword.
    Returns list of CVE summaries with ID, description, and CVSS score.

    Args:
        keyword: Search term (e.g. "SYN flood", "Log4j", "Apache", "SSH brute force").
        max_results: Maximum number of CVEs to return (default 5).
    """
    api_key = os.getenv("NVD_API_KEY") or None
    headers = {"apiKey": api_key} if api_key else {}

    try:
        resp = requests.get(
            NVD_BASE,
            params={"keywordSearch": keyword, "resultsPerPage": max_results},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for v in data.get("vulnerabilities", [])[:max_results]:
            cve = v.get("cve", {})
            desc = next(
                (d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"),
                "No description",
            )
            metrics = cve.get("metrics", {})
            score = None
            for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                if key in metrics and metrics[key]:
                    score = metrics[key][0].get("cvssData", {}).get("baseScore")
                    break
            results.append({
                "cve_id": cve.get("id"),
                "description": desc[:400],
                "cvss_score": score,
                "published": cve.get("published", "")[:10],
            })
        log.info("NVD lookup", keyword=keyword, n_results=len(results))
        return results
    except Exception as e:
        log.error("nvd_lookup error", error=str(e))
        return [{"error": str(e)}]


# Export tools list for graph
ALL_TOOLS = [ml_inference, rag_retrieval, nvd_lookup]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
