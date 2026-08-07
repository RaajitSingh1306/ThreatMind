# ThreatMind

**Cybersecurity Threat Intelligence Platform** — end-to-end ML + LLM agent for network intrusion detection, anomaly detection, and CVE-grounded threat reasoning.

Built to demonstrate production-grade ML engineering: data engineering, classical ML, deep learning, RAG, LLM agents, fine-tuning, AWS deployment, CI/CD, and model monitoring — in a single coherent system.

---

## What It Does

Given a network traffic record or a natural language query, ThreatMind:

1. **Classifies** the traffic as benign/attack + attack category (XGBoost/LightGBM)
2. **Detects** novel anomalies via reconstruction error (PyTorch Autoencoder)
3. **Explains** predictions with SHAP feature attributions
4. **Answers** threat questions grounded in CVE/NVD data via RAG
5. **Synthesizes** all three into a structured analyst report via a LangGraph agent

---

## Architecture Summary

```
CICIDS2017 Data → Airflow + PySpark → S3 (Parquet)
                        ↓
              sklearn Pipeline + XGBoost/LGB  ←→  MLflow
              PyTorch Autoencoder              ←→  MLflow
                        ↓
              FastAPI /predict + TorchServe
                        ↓
              LangGraph Agent
              ├── Tool: ML Inference (FastAPI)
              ├── Tool: RAG over CVE corpus (ChromaDB)
              └── Tool: NVD CVE lookup (REST API)
                        ↓
              OpenAI / Claude API (final synthesis)
                        ↓
              Deployed: EC2 + ECR + S3 + CloudWatch
```

Full diagram → [`SYSTEM_ARCHITECTURE.md`](./SYSTEM_ARCHITECTURE.md)

---

## Stack

| Layer | Technology |
|---|---|
| Data Engineering | Apache Airflow, PySpark, DuckDB, S3 (Parquet) |
| ML | scikit-learn, XGBoost, LightGBM, SHAP, MLflow |
| Deep Learning | PyTorch (Autoencoder), HuggingFace PEFT (LoRA) |
| RAG | ChromaDB, all-MiniLM-L6-v2, LangChain retriever |
| Agent | LangGraph, OpenAI/Anthropic API, NVD REST API |
| Evaluation | RAGAS, SHAP, calibration curves, AUC-PR |
| Serving | FastAPI, TorchServe, Docker |
| Cloud | AWS S3, EC2, ECR, CloudWatch |
| CI/CD | GitHub Actions |
| Observability | structlog, CloudWatch Logs + Metrics, drift detection |

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection dataset.
- ~2.8M network flow records
- 80 features (packet lengths, flow duration, flags, etc.)
- Labels: BENIGN, DoS, DDoS, PortScan, Brute Force, Web Attack, Infiltration, Botnet

Download: [UNB CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

Place CSVs in `data/raw/` or configure S3 path in `airflow/dags/ingest_cicids.py`.

---

## Quickstart

### Prerequisites
- Python 3.10+
- Docker + Docker Compose
- AWS CLI configured (`aws configure`)
- OpenAI or Anthropic API key

### Local Setup

```bash
git clone https://github.com/<your-handle>/threatmind.git
cd threatmind
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys, S3 bucket, AWS region
```

### Run Data Pipeline (local, DuckDB mode)

```bash
# Bypass Airflow for local dev — run transforms directly
python src/preprocessing/spark_transforms.py --input data/raw/ --output data/processed/
```

### Train Models

```bash
python src/ml/train.py --experiment-name threatmind-v1
# Logs to MLflow. View at http://localhost:5000 after: mlflow ui
```

### Build RAG Index

```bash
python src/rag/ingest.py --corpus data/cve_corpus/ --collection threatmind_cves
```

### Run API Locally

```bash
uvicorn src.serving.api:app --reload --port 8000
# Health check: http://localhost:8000/health
```

### Run Agent Query

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Is this traffic pattern consistent with a SYN flood? Flow duration: 0.002s, Fwd Packets: 1, Bwd Packets: 0"}'
```

---

## API Reference

### `POST /predict`
Classify a network flow record.

**Request:**
```json
{
  "features": {
    "flow_duration": 0.002,
    "fwd_packets": 1,
    "bwd_packets": 0,
    "fwd_packet_length_mean": 54.0,
    "...": "..."
  }
}
```

**Response:**
```json
{
  "prediction": "DoS",
  "confidence": 0.94,
  "anomaly_score": 0.73,
  "shap_top_features": [
    {"feature": "fwd_packets", "value": 1.0, "shap": 0.42},
    {"feature": "flow_duration", "value": 0.002, "shap": 0.31}
  ]
}
```

### `POST /agent/query`
Natural language threat query routed through the LangGraph agent.

**Request:**
```json
{"query": "What CVEs are associated with Apache Log4j RCE and how severe are they?"}
```

**Response:**
```json
{
  "answer": "...",
  "sources": ["CVE-2021-44228", "CVE-2021-45046"],
  "tool_calls": ["rag_retrieval", "nvd_lookup"],
  "reasoning_trace": "..."
}
```

---

## Project Structure

```
threatmind/
├── airflow/dags/          # Airflow pipeline DAGs
├── src/
│   ├── preprocessing/     # PySpark transforms, feature engineering
│   ├── ml/                # sklearn pipeline, XGBoost, autoencoder, eval
│   ├── rag/               # ingest, retriever, RAGAS eval
│   ├── agent/             # LangGraph graph, tools, prompts
│   ├── finetuning/        # LoRA config + training
│   └── serving/           # FastAPI app, TorchServe handler, schemas
├── monitoring/            # CloudWatch logger, drift detector
├── tests/                 # unit + integration
├── .github/workflows/     # CI/CD pipeline
├── docker/                # Dockerfiles for API and TorchServe
├── infra/                 # AWS bootstrap scripts
└── notebooks/             # EDA
```

---

## CI/CD Pipeline

On push to `main`:

1. **Lint** — `ruff check` + `black --check`
2. **Test** — `pytest tests/` (unit + integration)
3. **Build** — `docker build` API image
4. **Push** — push to ECR
5. **Deploy** — SSH into EC2, pull latest image, restart container

Pipeline config: [`.github/workflows/ci_cd.yml`](.github/workflows/ci_cd.yml)

---

## Model Performance

| Model | Metric | Value |
|---|---|---|
| XGBoost (multi-class) | AUC-PR | 0.93 |
| LightGBM (multi-class) | ROC-AUC | 0.97 |
| Autoencoder (anomaly) | F1 @ threshold | 0.86 |
| RAG pipeline | RAGAS faithfulness | 0.82 |
| LoRA threat classifier | F1 macro | 0.79 |

*Fill in actuals post-training.*

---

## Monitoring

- **Prediction distribution** tracked per 1-hour window → CloudWatch custom metric
- **Latency** (p50/p95/p99) per endpoint → CloudWatch
- **Error rate** → CloudWatch alarm (threshold: > 1% over 5 min)
- **Drift proxy**: KL divergence on prediction score distribution vs. training baseline

Logs shipped via `structlog` → CloudWatch Logs → queryable with CloudWatch Insights.

---

## Fine-Tuning (LoRA)

DistilBERT fine-tuned on threat category classification using NVD CWE descriptions as labels.

- Base model: `distilbert-base-uncased`
- PEFT method: LoRA (`rank=8`, `alpha=32`)
- Target modules: `q_lin`, `v_lin`
- Dataset: ~15K CVE description → CWE category pairs

```bash
python src/finetuning/train_lora.py --epochs 3 --output models/lora_threat_clf
```

---

## RAG Evaluation

```bash
python src/rag/eval_ragas.py --testset data/rag_testset.json
# Outputs: faithfulness, context_precision, context_recall, answer_relevancy
```

---

## AWS Infrastructure

| Resource | Purpose |
|---|---|
| S3 bucket | Raw data, Parquet, model artifacts, logs |
| EC2 (t3.medium) | FastAPI + TorchServe + MLflow |
| ECR | Docker image registry |
| CloudWatch | Logs, metrics, alarms |

Bootstrap:
```bash
bash infra/aws_setup.sh  # creates S3 bucket, ECR repo, EC2 instance
```

---

## Alignment with DS/ML Engineering JD Requirements

| Requirement | Implementation |
|---|---|
| End-to-end ML pipeline | Airflow + PySpark + sklearn + MLflow |
| CI/CD for ML | GitHub Actions: lint → test → build → deploy |
| Model serving frameworks | FastAPI (XGBoost) + TorchServe (PyTorch) |
| Monitoring + logging | CloudWatch + structlog + drift detection |
| GenAI / LLM integration | LangGraph agent + OpenAI/Anthropic API |
| RAG pipeline | ChromaDB + MiniLM + RAGAS eval |
| LLM fine-tuning (LoRA/QLoRA) | PEFT LoRA on DistilBERT |
| Cloud (AWS) | S3 + EC2 + ECR + CloudWatch |
| Performance + scalability | Docker, load tested with Locust |

---

## Background

Built as a portfolio project demonstrating production ML engineering skills, with domain grounding from work at the Data Security Council of India (DSCI) on national technology capability mapping and AI/cybersecurity research.

Dataset: CICIDS2017 (publicly available, UNB).
CVE data: National Vulnerability Database (NVD) public API.

---

## License

MIT
