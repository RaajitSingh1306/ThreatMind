# ThreatMind — Cybersecurity Threat Intelligence Platform

[![CI Pipeline](https://img.shields.io/badge/CI-lint%20%E2%86%92%20test%20%E2%86%92%20build%20%E2%86%92%20deploy-brightgreen)](#12-deployment)
[![Weighted F1](https://img.shields.io/badge/Weighted%20F1-0.97-emerald)](#8-results--evaluation)
[![Anomaly F1](https://img.shields.io/badge/Anomaly%20F1-0.86-blue)](#8-results--evaluation)
[![RAGAS Faithfulness](https://img.shields.io/badge/RAGAS%20Faithfulness-0.82-purple)](#8-results--evaluation)
[![AWS](https://img.shields.io/badge/Deploy%20on-AWS%20EC2%20%2B%20ECR-FF9900)](#12-deployment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#16-license--disclaimer)

| Endpoint | URL / Target |
|---|---|
| **Health Check** | `http://<EC2_IP>:8000/health` |
| **Interactive API Docs** | `http://<EC2_IP>:8000/docs` |
| **MLflow Registry** | `http://<EC2_IP>:5000` |

An end-to-end cybersecurity threat intelligence platform that unifies classical machine learning, deep learning anomaly detection, Retrieval-Augmented Generation (RAG), and autonomous LLM agents into an auditable production system. ThreatMind classifies network flows across 15 attack classes using an **XGBoost + LightGBM soft-voting ensemble** (Weighted F1: 0.97), detects novel/unseen zero-day intrusions via a **PyTorch Autoencoder** trained on benign traffic, delivers exact **SHAP TreeExplainer** feature attributions, grounds incident analysis in **NVD CVE vulnerability data via ChromaDB**, and synthesizes structured threat analyst reports via a **LangGraph ReAct agent**.

---

## Table of Contents

- [1. What This Project Does](#1-what-this-project-does)
- [2. Why It Was Built](#2-why-it-was-built)
- [3. System Architecture](#3-system-architecture)
- [4. Tech Stack & Libraries](#4-tech-stack--libraries)
- [5. Data](#5-data)
- [6. Step-by-Step Pipeline](#6-step-by-step-pipeline)
- [7. Problems Faced & How We Solved Them](#7-problems-faced--how-we-solved-them)
- [8. Results & Evaluation](#8-results--evaluation)
- [9. Project Structure](#9-project-structure)
- [10. Getting Started](#10-getting-started)
- [11. API Reference](#11-api-reference)
- [12. Deployment](#12-deployment)
- [13. Connected Portfolio Projects](#13-connected-portfolio-projects)
- [14. Limitations & Known Issues](#14-limitations--known-issues)
- [15. Roadmap / Future Expansion](#15-roadmap--future-expansion)
- [16. License & Disclaimer](#16-license--disclaimer)

---

## 1. What This Project Does

Given a raw network flow record or natural language security query, ThreatMind delivers:

- **15-Class Traffic Classification**: Classifies traffic as benign or one of 14 specific attack categories (DoS Hulk, PortScan, DDoS, Slowloris, Bot, Web Attacks, etc.) via a soft-voting ensemble.
- **Unsupervised Zero-Day Anomaly Detection**: Flags novel or unmodeled attack patterns via reconstruction error using a PyTorch Autoencoder trained exclusively on benign traffic.
- **Real-Time SHAP Feature Attribution**: Computes exact Shapley feature values ($O(TLD)$) explaining why the model classified a packet as malicious.
- **CVE & NVD Grounded Retrieval (RAG)**: Queries an indexed corpus of National Vulnerability Database (NVD) CVE records using ChromaDB to retrieve real-world exploit context and CVSS severity metrics.
- **Autonomous Multi-Tool Agent (LangGraph)**: Evaluates complex analyst prompts by autonomously chaining ML prediction, RAG retrieval, and live NVD API lookups into a structured incident report.
- **LoRA Fine-Tuned NLP Classifier**: Fine-tunes DistilBERT using Parameter-Efficient Fine-Tuning (PEFT/LoRA) to classify natural language threat descriptions into top-20 Common Weakness Enumeration (CWE) categories.

---

## 2. Why It Was Built

- **Alert Fatigue in Security Operations Centers (SOC)**: IDS/IPS systems trigger thousands of daily alerts; ~95% are benign false positives. Analysts waste hours manually triaging alerts with no ML-driven prioritization.
- **Siloed Threat Intelligence**: Flow telemetry, anomaly scoring, and CVE vulnerability databases exist in disconnected tools with no unified reasoning layer.
- **Black-Box Detection Frustration**: Traditional intrusion detection models provide binary flags without explanations, making it difficult to justify escalation decisions to incident response leads.
- **Inability to Flag Novel Zero-Days**: Static signature rulesets fail against zero-day exploits. The autoencoder reconstruction approach isolates deviations without requiring labeled attack samples.

---

## 3. System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                │
│                                                                        │
│   CICIDS2017 (~2.83M records) ──► Airflow DAG ──► PySpark/DuckDB       │
│   NVD CVE Ingestion           ──► Batch API   ──► ChromaDB Vector Store│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                              ML LAYER                                  │
│                                                                        │
│   Supervised Soft-Voting Ensemble     PyTorch Autoencoder (Unsupervised)│
│   ├── MedianImputer + StandardScaler  ├── FC(256→128→64→128→256→N)     │
│   ├── XGBoost (300 trees, depth=7)    ├── Trained on Benign-only flows │
│   ├── LightGBM (300 trees, depth=7)   ├── 95th percentile error cutoff │
│   └── SHAP TreeExplainer Attribution  └── Sigmoid normalized score     │
│                                                                        │
│   MLflow Experiment Tracking (models, parameters, PR curves, metrics)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           SERVING LAYER                                │
│                                                                        │
│   FastAPI Gateway (:8000)             TorchServe Model Engine          │
│   ├── POST /predict                   ├── High-throughput Autoencoder  │
│   ├── POST /agent/query               │   scoring                      │
│   └── GET  /health, /metrics          └── Lifespan model caching       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        RAG & AGENT LAYER                               │
│                                                                        │
│   LangGraph ReAct Agent (StateGraph loop: Reason ⇄ Act ⇄ Observe)       │
│   ├── Tool 1: ml_inference  ──► FastAPI /predict                       │
│   ├── Tool 2: rag_retrieval ──► ChromaDB + all-MiniLM-L6-v2 embeddings │
│   └── Tool 3: nvd_lookup    ──► NVD REST API v2.0                      │
│                                                                        │
│   LLM Backends: Groq (Llama 3.3 70B), Together AI, or local Ollama     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     INFRASTRUCTURE & CI/CD                             │
│                                                                        │
│   AWS Cloud: S3 (Lakehouse) + ECR (Containers) + EC2 (t3.medium API)   │
│   CI/CD: GitHub Actions (Lint → Test → Docker Build → ECR → EC2 Deploy)│
│   Observability: structlog → CloudWatch Logs + KL Divergence Drift     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tech Stack & Libraries

| Library / Tool | Version | Purpose | Rationale |
|---|---|---|---|
| **Python** | `>=3.11` | Core backend runtime | Modern asynchronous programming and ML stack support |
| **XGBoost** | `^2.0.3` | Gradient boosting classifier | High-performance multi-class tree classification with GPU/multithreading |
| **LightGBM** | `^4.3.0` | Gradient boosting classifier | Fast histogram-based tree learning complementing XGBoost diversity |
| **PyTorch** | `^2.2.0` | Deep learning autoencoder | Neural network framework for unsupervised bottleneck reconstruction |
| **scikit-learn** | `^1.4.0` | Pipeline & soft-voting | `VotingClassifier(voting='soft')`, imputation, and evaluation metrics |
| **SHAP** | `^0.44.0` | Explainability | `TreeExplainer` on XGBoost delivering exact sub-25ms Shapley values |
| **ChromaDB** | `^0.4.22` | Vector database | Lightweight, persistent local-first vector store for CVE/NVD chunk embeddings |
| **Sentence-Transformers**| `^2.3.1` | Dense embeddings | `all-MiniLM-L6-v2` generating 384-dim semantic embeddings |
| **LangGraph** | `^0.0.26` | Agent orchestration | Multi-tool ReAct state graph with explicit tool calling loops |
| **Groq / Together / Ollama** | Dynamic | LLM inference backends | High-speed, free-tier LLM providers for threat synthesis |
| **MLflow** | `^2.10.0` | Experiment registry | Centralized tracking for model parameters, artifacts, and PR curves |
| **Apache Airflow** | `^2.8.1` | Pipeline orchestration | Automated data download, validation, and training DAG workflows |
| **PySpark / DuckDB** | `^3.5.0` / `^0.9.2` | Data transformations | PySpark for production big data; DuckDB for lightweight local development |
| **PEFT / Transformers** | `^0.8.2` / `^4.37.2` | LoRA fine-tuning | Parameter-efficient fine-tuning on DistilBERT for threat text classification |
| **FastAPI** | `^0.109.0` | REST API gateway | High-throughput async API with Pydantic schemas and OpenAPI docs |
| **structlog** | `^24.1.0` | Structured logging | JSON structured logging routed to AWS CloudWatch |
| **pytest** | `^8.0.0` | Automated testing | Comprehensive unit and integration test suite |

---

## 5. Data

### CICIDS2017 Benchmark Dataset

- **Source**: Canadian Institute for Cybersecurity (UNB).
- **Scale**: ~2.83 Million network flow records.
- **Features**: 80 base features (packet lengths, flow duration, IAT, flags, byte counts) + 11 engineered features = **91 total features**.
- **Classes (15 Categories)**: BENIGN, DoS Hulk, PortScan, DDoS, DoS GoldenEye, FTP-Patator, SSH-Patator, DoS Slowloris, DoS Slowhttptest, Bot, Web Attack (Brute Force, XSS, SQL Injection), Infiltration, Heartbleed.
- **Engineered Higher-Order Features**:
  - `pkt_asymmetry`: $(Fwd - Bwd) / Total$ (highlights one-directional SYN floods).
  - `byte_asymmetry`: $(FwdBytes - BwdBytes) / Total$ (distinguishes exfiltration from injection).
  - `total_pkts_per_s`: $Total / Duration$ (identifies high-frequency packet bursts).
  - `syn_flag_count_ratio`: $SYN / Total$ (fingerprints pure SYN floods).
  - `iat_cv`: $\sigma_{IAT} / \mu_{IAT}$ (quantifies inter-arrival time uniformity; bots exhibit near-zero CV).
  - `fwd_header_ratio`: $FwdHeaderLen / FwdPayload$ (detects header-only probe scans).

### NVD CVE RAG Corpus

- **Source**: National Vulnerability Database (NVD) REST API v2.0 + local threat advisory JSON files.
- **Chunking**: 512 tokens (~2,048 characters) with 64-token overlap.
- **Embedding**: `all-MiniLM-L6-v2` (384-dimensional cosine vectors).
- **Indexed Entities**: CVE ID, CVSS score, severity tier, CWE mapping, and exploit descriptions.

---

## 6. Step-by-Step Pipeline

1. **Ingestion & Data Normalization (`airflow/dags/ingest_cicids.py`)**: Download raw CICIDS2017 CSVs; PySpark cleans headers to snake_case, handles NaNs, and converts labels.
2. **Feature Engineering (`src/preprocessing/feature_engineering.py`)**: Derive 11 higher-order asymmetric and packet density metrics; save partitioned `train.parquet` and `val.parquet`.
3. **Ensemble & Autoencoder Training (`src/ml/train.py`)**:
   - Fit `VotingClassifier` (XGBoost + LightGBM soft voting) on full feature matrix.
   - Train PyTorch Autoencoder exclusively on benign traffic; compute 95th percentile reconstruction threshold.
   - Log parameters, classification metrics, and models to MLflow (`mlflow.db`).
4. **CVE Knowledge Base Ingestion (`src/rag/ingest.py`)**: Query NVD API v2.0; chunk and embed CVE advisories into ChromaDB (`threatmind_cves`).
5. **LoRA Fine-Tuning (`src/finetuning/train_lora.py`)**: Fine-tune DistilBERT via PEFT/LoRA (rank=8, alpha=32) for text classification into top-20 CWE categories.
6. **Inference & Serving (`src/serving/api.py`)**: Expose endpoints via FastAPI: `/predict` for classification + anomaly scoring + SHAP, and `/agent/query` for autonomous LangGraph ReAct investigations.

---

## 7. Problems Faced & How We Solved Them

| Problem | Impact | How We Got Around It |
|---|---|---|
| **Extreme class imbalance** (BENIGN = 80% of 2.83M records) | Model overwhelmingly predicted BENIGN; minority attack classes (SSH-Patator, Bot, Heartbleed) had near-zero recall | Used a **soft-voting ensemble** (XGBoost + LightGBM) which averages class probabilities, significantly reducing individual model bias compared to hard voting. Documented minority class gaps and scheduled SMOTE + focal loss in roadmap |
| **80 raw features with inconsistent column names** | Raw CICIDS2017 CSVs had leading/trailing spaces, mixed casing, and special characters in column headers | Built `spark_transforms.py` to **programmatically rename all 80 columns to snake_case**, cast to numeric types, and replace `inf`/`NaN` values in a single PySpark/DuckDB vectorized pass |
| **NVD API rate limiting** (5 req/30s without API key) | Building a multi-year CVE corpus for RAG took ~6 hours per year of CVE data due to throttling | Implemented **dual ingestion mode**: NVD API for fresh delta syncs, and local `data/cve_corpus/` JSON files for pre-downloaded historical dumps, bypassing rate limits entirely |
| **Novel / unseen attack detection without labels** | Signature-based classifiers cannot flag zero-day attacks that do not match historical training labels | Trained a **PyTorch Autoencoder exclusively on benign traffic** — novel attacks are flagged by high reconstruction error. No labeled attack samples are needed for zero-day detection |
| **Autoencoder threshold selection** | A too-low threshold produced floods of false positive alerts; a too-high threshold missed subtle intrusions | Set threshold at the **95th percentile of training reconstruction errors**, balancing sensitivity vs. false positive rate. Normalized anomaly scores via sigmoid: $\text{score} = 1 / (1 + \exp(-(e/\theta - 1)))$ |
| **SHAP on ensemble was computationally slow** | Computing KernelSHAP on the full VotingClassifier took multiple seconds per request | Used `shap.TreeExplainer` on the **XGBoost sub-model only**, which delivers exact $O(TLD)$ polynomial-time Shapley values — fast enough for real-time per-request API explanations |
| **PySpark dependency too heavy for local dev** (~1GB) | Local development required installing full Java/Hadoop Spark stacks just to run transforms | Implemented a **DuckDB fallback** (`DuckDB for local dev, PySpark for production`) that runs identical SQL transforms without JVM overhead, controlled via an environment flag |

---

## 8. Results & Evaluation

### Multi-Class Classification Performance (CICIDS2017 Benchmark)

| Model Architecture | Weighted F1 | Micro Avg F1 | Weighted Precision | Weighted Recall |
|---|:---:|:---:|:---:|:---:|
| **XGBoost + LightGBM Ensemble** | **0.97** | **0.98** | **0.97** | **0.98** |

### Per-Class Performance Breakdown

| Attack Category | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|---:|
| **BENIGN** | **0.99** | **0.99** | **0.99** | 454,266 |
| **DoS Hulk** | **0.94** | **0.95** | **0.95** | 45,923 |
| **PortScan** | **0.98** | **0.98** | **0.98** | 31,672 |
| **DDoS** | **0.94** | **0.97** | **0.95** | 25,806 |
| **DoS GoldenEye** | 0.86 | 0.24 | 0.38 | 2,055 |
| **DoS Slowhttptest** | 0.57 | 0.51 | 0.54 | 1,104 |

### Unsupervised Anomaly Detection

- **PyTorch Autoencoder**: Optimal Threshold F1: **0.86** on validation intrusion data.

### RAG & Fine-Tuning Quality Benchmarks

- **RAGAS Faithfulness**: **0.82** (grounded in NVD CVE context).
- **LoRA DistilBERT Threat Classifier**: Macro F1: **0.79** across top-20 CWE categories.

### Automated Test Suite

- Unit tests (`tests/unit/`): Pipeline build/fit, Autoencoder forward pass, Agent tool routing.
- Integration tests (`tests/integration/`): API endpoints, schemas, CORS, and error handling.

---

## 9. Project Structure

```text
threatmind/
├── airflow/
│   └── dags/
│       ├── ingest_cicids.py        # Airflow DAG: download → validate → PySpark clean → Parquet
│       └── train_pipeline.py       # Airflow DAG: load Parquet → train models → log to MLflow
├── src/
│   ├── preprocessing/
│   │   ├── spark_transforms.py     # PySpark/DuckDB header cleanup and type-casting
│   │   └── feature_engineering.py  # 11 derived packet asymmetry and density features
│   ├── ml/
│   │   ├── pipeline.py             # sklearn Pipeline: VotingClassifier (XGB+LGB) + SHAP
│   │   ├── autoencoder.py          # PyTorch Autoencoder + AnomalyDetector wrapper
│   │   ├── evaluation.py           # AUC-PR, F1, classification reports, PR curves
│   │   └── train.py                # MLflow entry point
│   ├── rag/
│   │   ├── ingest.py               # NVD API fetch + local corpus → ChromaDB upsert
│   │   ├── retriever.py            # ThreatRAG: ChromaDB retrieval + LLM synthesis
│   │   └── eval_ragas.py           # RAGAS evaluation metrics
│   ├── agent/
│   │   ├── graph.py                # LangGraph StateGraph ReAct loop
│   │   ├── tools.py                # Tools: ml_inference, rag_retrieval, nvd_lookup
│   │   └── prompts.py              # System prompts & chain-of-thought templates
│   ├── finetuning/
│   │   ├── lora_config.py          # LoRA hyperparameters + top-20 CWE label map
│   │   └── train_lora.py           # PEFT LoRA fine-tuning on DistilBERT
│   └── serving/
│       ├── api.py                  # FastAPI: /predict, /agent/query, /health, /metrics
│       ├── schemas.py              # Pydantic request and response schemas
│       └── torchserve_handler.py   # TorchServe custom handler
├── monitoring/
│   ├── cloudwatch_logger.py        # structlog → CloudWatch Logs
│   └── drift_detector.py           # KL divergence prediction distribution drift detector
├── tests/
│   ├── unit/                       # Unit tests
│   └── integration/                # API integration tests
├── docker/
│   ├── Dockerfile.api              # Multi-stage Python 3.12 container
│   └── Dockerfile.torchserve       # TorchServe container
├── infra/
│   └── aws_setup.sh                # AWS S3, ECR, EC2 bootstrap script
├── docker-compose.yml              # Local multi-service orchestration
├── requirements.txt                # Production dependencies
└── README.md                       # Project documentation
```

---

## 10. Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- LLM Provider API Key (Groq, Together AI, or local Ollama)

### 1. Environment Setup

```bash
cd "ThreatMind"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your LLM_PROVIDER and API key
```

### 2. Preprocess Data (DuckDB Local Mode)

```bash
python src/preprocessing/spark_transforms.py --input archive/ --output data/processed/
```

### 3. Train Models & Track via MLflow

```bash
python src/ml/train.py --experiment-name threatmind-v1
mlflow ui --port 5000
```

### 4. Build RAG Index

```bash
# Ingest from local corpus
python src/rag/ingest.py --corpus-dir data/cve_corpus/ --collection threatmind_cves
```

### 5. Launch FastAPI Backend

```bash
uvicorn src.serving.api:app --reload --port 8000
```

- Swagger UI: `http://localhost:8000/docs`

---

## 11. API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service liveness and model loading status |
| `GET` | `/metrics` | Operational latency metrics (p50/p95/p99) |
| `POST` | `/predict` | Multi-class classification + Autoencoder anomaly score + SHAP |
| `POST` | `/agent/query` | LangGraph ReAct agent threat investigation |

### Sample Payload (`POST /predict`)

```json
{
  "features": {
    "flow_duration": 0.002,
    "total_fwd_packets": 1,
    "total_backward_packets": 0,
    "pkt_asymmetry": 1.0,
    "syn_flag_count_ratio": 1.0
  }
}
```

### Sample Response (`POST /predict`)

```json
{
  "prediction": "DoS",
  "confidence": 0.94,
  "anomaly_score": 0.73,
  "is_anomaly": true,
  "top_features": [
    {"feature": "pkt_asymmetry", "attribution": 0.38},
    {"feature": "syn_flag_count_ratio", "attribution": 0.29}
  ]
}
```

---

## 12. Deployment

### AWS Cloud Architecture
- **AWS S3**: Storage lake for raw CICIDS2017 Parquet and model checkpoints.
- **AWS ECR**: Docker image registry for multi-stage API containers.
- **AWS EC2 (t3.medium)**: Production API host managed with systemd or Docker.
- **AWS CloudWatch**: Structured JSON logs and KL divergence distribution drift alerts.

### GitHub Actions CI/CD Pipeline
The repository features a 5-stage automated GitHub Actions workflow (`.github/workflows/ci_cd.yml`):
`Lint (ruff)` $\rightarrow$ `Test (pytest)` $\rightarrow$ `Docker Build` $\rightarrow$ `Push ECR` $\rightarrow$ `Deploy EC2`.

---

## 13. Connected Portfolio Projects

- **[Credit Default Predictor](https://github.com/RaajitSingh1306/Credit-Default-Predictor)**: Machine learning classification pipeline using TreeSHAP explainability.
- **[NSEI Daily Stock Pipeline](https://github.com/RaajitSingh1306/NSEI-Daily-Stock-Pipeline)**: Production Apache Airflow and PySpark lakehouse pipeline architecture.
- **[SEBI RAG Bot](https://github.com/RaajitSingh1306/sebi-rag-bot)**: Multi-agent system utilizing LangGraph and Qdrant vector retrieval.
- **[Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform)**: Time-series machine learning platform with walk-forward CV and MLflow tracking.

---

## 14. Limitations & Known Issues

- **Minority Attack Recall**: Rare attack types (Heartbleed, Infiltration, Web Attacks) have low sample representations in CICIDS2017, yielding lower minority recall.
- **NVD API Rate Limits**: Unauthenticated queries to the NVD API are throttled to 5 requests per 30 seconds.
- **Fixed Anomaly Cutoff**: The 95th percentile reconstruction error threshold is fixed; dynamic regime-adaptive cutoffs are not yet implemented.
- **Single-Turn Agent Conversations**: The LangGraph ReAct agent evaluates queries independently; multi-turn persistent incident tracking is currently stateless.

---

## 15. Roadmap / Future Expansion

- [ ] **SMOTE / Focal Loss Implementation**: Rebalance minority attack distributions to improve recall on rare zero-day exploits.
- [ ] **Streaming Kafka Ingestion**: Process live Zeek / Suricata network flows via Apache Kafka.
- [ ] **MITRE ATT&CK Knowledge Graph**: Construct graph relationships linking CVE vulnerabilities to ATT&CK tactics, techniques, and procedures (TTPs).
- [ ] **Multi-Turn Incident Investigation**: Add persistent session memory for continuous SOC analyst investigations.

---

## 16. License & Disclaimer

### License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Disclaimer
This software is intended strictly for cybersecurity defense research, educational modeling, and incident triage demonstration. It must not be utilized for malicious activities or unauthorized network intrusion.
