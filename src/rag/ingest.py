"""
RAG corpus ingestion for ThreatMind.

Downloads CVE data from NVD public API (free, no key required for basic use)
then chunks, embeds, and stores in ChromaDB.

Usage:
    python src/rag/ingest.py --year 2023 --year 2024 --collection threatmind_cves
    python src/rag/ingest.py --corpus-dir data/cve_corpus/ --collection threatmind_cves
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import chromadb
import requests
import structlog
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512  # tokens (approximate via chars / 4)
CHUNK_OVERLAP = 64
NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks (approx token-equivalent)."""
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    chunks = []
    start = 0
    while start < len(text):
        end = start + char_size
        chunks.append(text[start:end])
        start += char_size - char_overlap
    return [c.strip() for c in chunks if c.strip()]


def fetch_nvd_cves(year: int, api_key: str | None = None) -> list[dict]:
    """
    Fetch all CVEs for a given year from NVD API v2.
    Rate limit: 5 req/30s without key, 50 req/30s with key.
    """
    cves = []
    start_index = 0
    results_per_page = 2000
    headers = {"apiKey": api_key} if api_key else {}

    log.info("Fetching CVEs from NVD", year=year)

    while True:
        params = {
            "pubStartDate": f"{year}-01-01T00:00:00.000",
            "pubEndDate": f"{year}-12-31T23:59:59.999",
            "startIndex": start_index,
            "resultsPerPage": results_per_page,
        }
        try:
            resp = requests.get(NVD_BASE, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error("NVD fetch error", error=str(e))
            break

        vulnerabilities = data.get("vulnerabilities", [])
        cves.extend(vulnerabilities)
        total = data.get("totalResults", 0)
        log.info("NVD progress", year=year, fetched=len(cves), total=total)

        if len(cves) >= total:
            break

        start_index += results_per_page
        time.sleep(6 if not api_key else 0.6)  # rate limit

    return cves


def _extract_cve_text(vuln: dict) -> tuple[str, dict]:
    """Extract text description and metadata from NVD vulnerability dict."""
    cve = vuln.get("cve", {})
    cve_id = cve.get("id", "UNKNOWN")

    # Description
    descriptions = cve.get("descriptions", [])
    desc = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")

    # CVSS score
    metrics = cve.get("metrics", {})
    score = None
    severity = None
    for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
        if key in metrics and metrics[key]:
            m = metrics[key][0]
            score = m.get("cvssData", {}).get("baseScore")
            severity = m.get("cvssData", {}).get("baseSeverity")
            break

    # CWEs
    weaknesses = cve.get("weaknesses", [])
    cwes = []
    for w in weaknesses:
        for d in w.get("description", []):
            if d.get("lang") == "en":
                cwes.append(d["value"])

    # Published date
    published = cve.get("published", "")

    text = f"CVE ID: {cve_id}\nDescription: {desc}"
    if cwes:
        text += f"\nWeaknesses: {', '.join(cwes)}"
    if score:
        text += f"\nCVSS Score: {score} ({severity})"

    metadata = {
        "cve_id": cve_id,
        "score": score or 0.0,
        "severity": severity or "UNKNOWN",
        "cwes": json.dumps(cwes),
        "published": published[:10] if published else "",
    }
    return text, metadata


def ingest_from_nvd(years: list[int], collection_name: str, persist_dir: str, api_key: str | None = None) -> int:
    """Download CVEs from NVD and ingest into ChromaDB."""
    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(collection_name, embedding_function=embed_fn)

    total_chunks = 0
    for year in years:
        cves = fetch_nvd_cves(year, api_key)
        log.info("Processing year", year=year, n_cves=len(cves))

        docs, metas, ids = [], [], []
        for vuln in cves:
            text, metadata = _extract_cve_text(vuln)
            chunks = _chunk_text(text)
            for i, chunk in enumerate(chunks):
                doc_id = f"{metadata['cve_id']}_chunk{i}"
                docs.append(chunk)
                metas.append(metadata)
                ids.append(doc_id)

        # Batch upsert (ChromaDB supports batches of 5000)
        batch_size = 500
        for i in range(0, len(docs), batch_size):
            collection.upsert(
                documents=docs[i : i + batch_size],
                metadatas=metas[i : i + batch_size],
                ids=ids[i : i + batch_size],
            )

        total_chunks += len(docs)
        log.info("Ingested year", year=year, chunks=len(docs), total=total_chunks)

    return total_chunks


def ingest_from_dir(corpus_dir: str, collection_name: str, persist_dir: str) -> int:
    """Ingest local JSON/text files from a directory into ChromaDB with rich metadata."""
    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(collection_name, embedding_function=embed_fn)

    corpus_path = Path(corpus_dir)
    files = list(corpus_path.glob("*.json")) + list(corpus_path.glob("*.txt"))
    log.info("Ingesting local corpus", files=len(files), dir=corpus_dir)

    total = 0
    docs, metas, ids = [], [], []

    for f in files:
        if f.suffix.lower() == ".json":
            try:
                data = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, list):
                    for item in data:
                        cve_id = item.get("cve_id", f.stem)
                        title = item.get("title", "")
                        desc = item.get("description", "")
                        mitigation = item.get("mitigation", "")
                        cwes = item.get("cwes", [])
                        score = item.get("cvss_score", 0.0)
                        severity = item.get("severity", "UNKNOWN")
                        attack_type = item.get("attack_type", "")

                        doc_text = f"Identifier: {cve_id}\nTitle: {title}\nAttack Type: {attack_type}\nDescription: {desc}"
                        if cwes:
                            doc_text += f"\nWeaknesses: {', '.join(cwes)}"
                        if score:
                            doc_text += f"\nCVSS Score: {score} ({severity})"
                        if mitigation:
                            doc_text += f"\nMitigation & Remediation: {mitigation}"

                        chunks = _chunk_text(doc_text)
                        for idx, chunk in enumerate(chunks):
                            docs.append(chunk)
                            ids.append(f"{cve_id}_chunk{idx}_{len(ids)}")
                            metas.append({
                                "cve_id": cve_id,
                                "score": float(score or 0.0),
                                "severity": str(severity or "UNKNOWN"),
                                "cwes": json.dumps(cwes),
                                "attack_type": str(attack_type),
                                "source": f.name,
                            })
                    continue
            except Exception as e:
                log.warning("Failed structured JSON parse, falling back to raw text", file=f.name, error=str(e))

        # Fallback for plain text or unstructured JSON
        text = f.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            ids.append(f"{f.stem}_chunk{i}_{len(ids)}")
            metas.append({"source": f.name, "cve_id": f.stem, "score": 0.0, "severity": "UNKNOWN", "cwes": "[]"})

    # Batch upsert
    batch_size = 500
    for i in range(0, len(docs), batch_size):
        collection.upsert(
            documents=docs[i : i + batch_size],
            metadatas=metas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )
    total = len(docs)
    log.info("Local corpus ingested", total_chunks=total)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="ThreatMind RAG ingestion")
    parser.add_argument("--year", type=int, action="append", dest="years", help="NVD year to fetch (repeatable)")
    parser.add_argument("--corpus-dir", default=None, help="Local corpus directory to ingest")
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "threatmind_cves"))
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))
    args = parser.parse_args()

    api_key = os.getenv("NVD_API_KEY") or None

    if args.years:
        n = ingest_from_nvd(args.years, args.collection, args.persist_dir, api_key)
        log.info("NVD ingest complete", total_chunks=n)

    if args.corpus_dir:
        n = ingest_from_dir(args.corpus_dir, args.collection, args.persist_dir)
        log.info("Local corpus ingest complete", total_chunks=n)

    if not args.years and not args.corpus_dir:
        log.info("No source specified — fetching 2023+2024 from NVD by default")
        ingest_from_nvd([2023, 2024], args.collection, args.persist_dir, api_key)


if __name__ == "__main__":
    main()
