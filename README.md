# ThreatMind — Cybersecurity Threat Intelligence Platform

[![CI Pipeline](https://img.shields.io/badge/CI-lint%20→%20test%20→%20build%20→%20deploy-brightgreen)](#cicd-pipeline)
[![Weighted F1](https://img.shields.io/badge/Weighted%20F1-0.97-emerald)](#evaluation--verification)
[![RAGAS Faithfulness](https://img.shields.io/badge/RAGAS%20Faithfulness-0.82-blue)](#evaluation--verification)
[![AWS](https://img.shields.io/badge/Deploy%20on-AWS%20EC2%20+%20ECR-FF9900)](#deployment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

| | |
|---|---|
| **Health** | `http://<EC2_IP>:8000/health` |
| **API Docs** | `http://<EC2_IP>:8000/docs` |
| **MLflow** | `http://<EC2_IP>:5000` |

---

## What

ThreatMind is an **end-to-end cybersecurity threat intelligence platform** that combines classical ML, deep learning, RAG, and LLM agents into a single production-grade system. Given a network traffic record or a natural language threat query, ThreatMind:

| Capability | How |
|---|---|
| **Classifies** network traffic as benign or one of 14 attack categories | XGBoost + LightGBM soft-voting ensemble via sklearn Pipeline |
| **Detects** novel/unseen attack patterns via unsupervised anomaly detection | PyTorch Autoencoder trained on benign-only traffic, threshold on reconstruction error |
| **Explains** every prediction with feature-level attributions | SHAP TreeExplainer on the XGBoost sub-model |
| **Answers** threat intelligence questions grounded in CVE/NVD data | ChromaDB vector store + `all-MiniLM-L6-v2` embeddings + LLM synthesis |
| **Synthesises** all signals into a structured analyst report | LangGraph ReAct agent with 3 tools (ML inference, RAG retrieval, NVD lookup) |

**Example interaction:**

> **Analyst:** _"Is this traffic pattern consistent with a SYN flood? Flow duration: 0.002s, Fwd Packets: 1, Bwd Packets: 0"_
>
> **ThreatMind:**
> 1. **Threat Assessment:** Classified as **DoS** (confidence: 0.94). Anomaly score: 0.73 (above threshold).
> 2. **Severity:** HIGH — single-packet, zero-response flow with 100% packet asymmetry is a classic SYN flood signature.
> 3. **Relevant CVEs:** CVE-2019-11477 (SACK Panic), CVE-2018-5390 (SegmentSmack) — kernel-level SYN/ACK handling vulnerabilities.
> 4. **Recommended Actions:** Enable SYN cookies, rate-limit half-open connections, deploy IDS rules for asymmetric flow patterns.
> 5. **Confidence:** High — ML prediction aligns with anomaly detection and CVE context.

---

## Why

### The Problem

Security Operations Center (SOC) analysts face compounding challenges in modern threat detection:

1. **Alert fatigue** — IDS/IPS systems generate thousands of alerts daily; ~95% are false positives, and analysts manually triage each one with no ML-driven prioritisation
2. **Siloed intelligence** — Network traffic classification, anomaly detection, and CVE databases live in entirely separate tools with no unified reasoning layer
3. **Black-box predictions** — Existing ML-based intrusion detection systems output labels with no explanation, making it impossible to justify escalation decisions to management
4. **Static rulesets** — Signature-based detection cannot flag novel (zero-day) attack patterns that deviate from known signatures
5. **No conversational interface** — Analysts cannot ask "what CVEs are related to this traffic pattern?" — they must manually cross-reference NVD, MITRE ATT&CK, and vendor advisories

### The Solution

ThreatMind solves all five problems in a single coherent system:

- **Ensemble ML classification** (XGBoost + LightGBM) provides high-accuracy attack categorisation across 15 traffic classes, with soft-voting to reduce individual model bias
- **Unsupervised anomaly detection** (PyTorch Autoencoder) trains exclusively on benign traffic and flags novel attack patterns by reconstruction error — no labelled attack samples required
- **SHAP explainability** accompanies every prediction with the top contributing features and their attribution values, enabling auditable escalation decisions
- **RAG over CVE/NVD data** grounds every threat intelligence answer in real vulnerability data — the LLM can only synthesise from retrieved context, never from training data
- **LangGraph ReAct agent** autonomously decides which tools to invoke (ML inference, RAG retrieval, NVD API lookup), chains their outputs, and synthesises a structured analyst report

### Who Is This For

- **SOC analysts** who need automated triage with explainable, CVE-grounded threat assessments
- **Security engineers** building ML-augmented intrusion detection pipelines
- **ML engineers** seeking a production-grade reference architecture covering data engineering → training → serving → monitoring → CI/CD
- **Students and researchers** studying applied ML in cybersecurity

---

## How

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                  │
│                                                                      │
│   CICIDS2017 CSVs ──► Airflow DAG ──► PySpark/DuckDB ──► Parquet   │
│   NVD CVE JSON  ──────────────────────────────────────► ChromaDB    │
│                              │                                       │
│                           DuckDB (local dev fallback)                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                          ML LAYER                                    │
│                                                                      │
│   sklearn Pipeline                    PyTorch Autoencoder            │
│   ├─ MedianImputer                    ├─ Encoder: FC(256→128→64)    │
│   ├─ StandardScaler                   ├─ BatchNorm + Dropout(0.2)   │
│   ├─ VotingClassifier(soft)           ├─ Decoder: FC(64→128→256→N)  │
│   │  ├─ XGBoost (300 trees, d=7)     ├─ MSE reconstruction error   │
│   │  └─ LightGBM (300 trees, d=7)    └─ 95th percentile threshold  │
│   └─ SHAP TreeExplainer                                             │
│                                                                      │
│   MLflow Tracking Server ◄──── all experiments, models, metrics      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        SERVING LAYER                                 │
│                                                                      │
│   FastAPI (gateway)                   TorchServe                     │
│   ├─ POST /predict         ◄─────────► PyTorch autoencoder          │
│   ├─ POST /agent/query                                               │
│   ├─ GET  /health                                                    │
│   └─ GET  /metrics                                                   │
│                                                                      │
│   Docker (multi-stage) ──► ECR ──► EC2                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                     RAG + AGENT LAYER                                │
│                                                                      │
│   LangGraph ReAct Agent (reason → act → observe → repeat)           │
│   ├─ Tool 1: ml_inference  ──► FastAPI /predict                      │
│   ├─ Tool 2: rag_retrieval ──► ChromaDB → LLM synthesis             │
│   └─ Tool 3: nvd_lookup    ──► NVD REST API v2.0                    │
│                                                                      │
│   RAG Pipeline                                                       │
│   ├─ Corpus: NVD CVE JSON + threat intelligence reports              │
│   ├─ Chunking: 512 tokens, 64 overlap (char-approximated)           │
│   ├─ Embedding: all-MiniLM-L6-v2 (SentenceTransformers)             │
│   ├─ Vector Store: ChromaDB (persistent, local-first)                │
│   └─ LLM: Groq / Together AI / Ollama (all free-tier)               │
│                                                                      │
│   Evaluation: RAGAS (faithfulness, context recall, answer relevancy) │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      FINE-TUNING MODULE                              │
│                                                                      │
│   Base: DistilBERT (distilbert-base-uncased)                         │
│   Task: Threat category classification (top-20 CWE categories)       │
│   Method: LoRA via PEFT (rank=8, alpha=32, target: q_lin, v_lin)    │
│   Trainer: HuggingFace Trainer                                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                             │
│                                                                      │
│   AWS: S3 (data/models) + EC2 (serving) + ECR (images) + CloudWatch │
│   CI/CD: GitHub Actions (lint → test → build → push ECR → deploy)   │
│   Monitoring: structlog → CloudWatch Logs + custom metrics + drift   │
└─────────────────────────────────────────────────────────────────────┘
```

### How It Works Step-by-Step

**1. Data Ingestion & Feature Engineering (Airflow + PySpark/DuckDB)**

The Airflow DAG (`ingest_cicids.py`) downloads CICIDS2017 CSVs, validates schema, and triggers the PySpark/DuckDB preprocessing pipeline. The `spark_transforms.py` module renames all 80 raw columns to snake_case, casts to numeric types, replaces `inf`/`NaN`, and encodes 15 attack labels to integers. The `feature_engineering.py` module then derives 11 higher-order features:

| Engineered Feature | Formula | Why It's Needed |
|---|---|---|
| `pkt_asymmetry` | (fwd − bwd) / total | Captures one-directional flooding (SYN flood → asymmetry ≈ 1.0) |
| `byte_asymmetry` | (fwd_bytes − bwd_bytes) / total | Distinguishes exfiltration (high bwd) from injection (high fwd) |
| `total_pkts_per_s` | total_pkts / duration(s) | Burst rate — DDoS floods spike to millions pkt/s |
| `syn_flag_count_ratio` | SYN_flags / total_pkts | SYN flood fingerprint — ratio ≈ 1.0 for pure SYN flooding |
| `iat_cv` | IAT_std / IAT_mean | Inter-arrival burstiness — bots produce near-zero CV (uniform) |
| `fwd_header_ratio` | fwd_header_len / fwd_payload | High ratio signals header-only probes (PortScan, Heartbleed) |

The pipeline outputs `train.parquet` and `val.parquet` (80/20 stratified split, ~2.8M records total).

**2. Model Training (MLflow-tracked)**

Two model families are trained and logged to MLflow:

| Model | Training Data | Objective | MLflow Artifacts |
|---|---|---|---|
| **sklearn Pipeline** (XGBoost + LightGBM) | Full train set (80+ features) | Multi-class classification (15 labels) | `pipeline.joblib`, `feature_names.json`, classification report |
| **PyTorch Autoencoder** | Benign-only train set | Unsupervised anomaly detection via reconstruction error | `autoencoder.pt`, PR curve, threshold value |

The autoencoder architecture is: `Input(N) → FC(256) → BN → ReLU → Dropout(0.2) → FC(128) → BN → ReLU → FC(64) [bottleneck] → FC(128) → BN → ReLU → FC(256) → BN → ReLU → Dropout(0.2) → FC(N)`. The anomaly threshold is set at the 95th percentile of training reconstruction errors, and anomaly scores are normalised via sigmoid: `score = 1 / (1 + exp(-(error/threshold - 1)))`.

**3. RAG Corpus Ingestion**

The `src/rag/ingest.py` module supports two ingestion modes:

| Mode | Source | Rate Limiting |
|---|---|---|
| **NVD API** | Fetches CVEs by year from NVD REST API v2.0 | 5 req/30s (no key) or 50 req/30s (with key) |
| **Local corpus** | Reads `.json`/`.txt` files from `data/cve_corpus/` | No limit |

CVE descriptions, CWE IDs, CVSS scores, and severity levels are extracted, chunked (512 tokens, 64 overlap), embedded with `all-MiniLM-L6-v2`, and upserted into ChromaDB in batches of 500.

**4. Inference — ML Prediction (FastAPI `/predict`)**

For classification requests, the API loads the sklearn pipeline and autoencoder at startup. A feature vector is constructed from the request's feature dictionary, run through the pipeline (impute → scale → predict), and SHAP TreeExplainer computes per-feature attributions on the XGBoost sub-model. The autoencoder simultaneously scores the sample for novelty. The response includes: predicted label, confidence, anomaly score, anomaly flag, and top-10 SHAP features.

**5. Inference — Agent Query (FastAPI `/agent/query`)**

For natural language queries, the LangGraph ReAct agent executes:

```
User Query
    │
    ▼
LangGraph Agent (ReAct loop)
    ├── Decide: needs ML prediction?
    │       └──► Tool: ml_inference → POST /predict → XGBoost + SHAP
    │
    ├── Decide: needs threat context?
    │       └──► Tool: rag_retrieval → ChromaDB top-5 → LLM synthesis
    │
    └── Decide: needs CVE data?
            └──► Tool: nvd_lookup → NVD API keyword search → CVE metadata
    │
    ▼
LLM final synthesis (Groq / Together / Ollama)
    │
    ▼
Structured JSON response with: answer, sources, tool_calls, reasoning_trace
```

The agent supports three free LLM backends via `LLM_PROVIDER` env var:

| Provider | Model | Speed | Cost |
|---|---|---|---|
| **Groq** (recommended) | `llama-3.3-70b-versatile` | ~500 tok/s | Free, no credit card |
| **Together AI** | `Llama-3-70b-chat-hf` | ~200 tok/s | Free tier |
| **Ollama** | `llama3.2` (local) | Hardware-dependent | Fully local, no API key |

### Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Ensemble strategy** | Soft-voting (XGBoost + LightGBM) | Combines gradient boosting diversity; soft voting averages class probabilities for calibrated confidence |
| **Anomaly detection** | PyTorch Autoencoder (benign-only training) | Unsupervised — no labelled attack samples required for novel attack detection; reconstruction error naturally separates seen vs. unseen distributions |
| **Anomaly threshold** | 95th percentile of training MSE | Balances sensitivity vs. false positive rate; empirically validated on validation set |
| **SHAP integration** | TreeExplainer on XGBoost sub-model | O(TLD) complexity for tree-based models — fast enough for real-time per-request explanations |
| **Vector store** | ChromaDB (persistent, local-first) | Zero infrastructure overhead vs. Pinecone/Qdrant Cloud; data stays on-instance |
| **Embedding model** | `all-MiniLM-L6-v2` | Free, fast, 384-dim — sufficient accuracy for CVE domain at minimal memory cost |
| **Agent framework** | LangGraph `StateGraph` (ReAct loop) | Explicit graph with typed state and conditional edges — cleaner than ad-hoc LangChain chains for multi-tool orchestration |
| **LLM provider** | Groq (default) with Together/Ollama fallbacks | Zero-cost inference; provider-agnostic via `LLM_PROVIDER` env var |
| **Deployment target** | EC2 (not Lambda) | Model loading latency (~5s for XGBoost + PyTorch) makes Lambda cold starts unacceptable for inference |
| **Fine-tuning method** | LoRA via PEFT (rank=8, alpha=32) | Parameter-efficient — only 0.3% of DistilBERT parameters trained; full fine-tune unnecessary at this dataset scale |
| **Drift detection** | KL divergence on prediction score distribution | Simple, interpretable proxy that works without ground-truth labels in production |
| **Data fallback** | DuckDB for local dev, PySpark for production | DuckDB avoids ~1 GB PySpark dependency for local development while maintaining identical transforms |

---

## Evaluation & Verification

### Classification Performance

Trained and evaluated on CICIDS2017 (~2.8M records, 80/20 split):

| Model | Metric | Value | Status |
|---|---|---|:---:|
| **XGBoost + LightGBM** (ensemble) | Weighted F1 | **0.97** | ✅ |
| **XGBoost + LightGBM** (ensemble) | Micro Avg F1 | **0.98** | ✅ |
| **XGBoost + LightGBM** (ensemble) | Weighted Precision | **0.97** | ✅ |
| **XGBoost + LightGBM** (ensemble) | Weighted Recall | **0.98** | ✅ |

#### Per-Class Breakdown

| Attack Category | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|---:|
| BENIGN | 0.99 | 0.99 | 0.99 | 454,266 |
| DoS Hulk | 0.94 | 0.95 | 0.95 | 45,923 |
| PortScan | 0.98 | 0.98 | 0.98 | 31,672 |
| DDoS | 0.94 | 0.97 | 0.95 | 25,806 |
| DoS GoldenEye | 0.86 | 0.24 | 0.38 | 2,055 |
| DoS Slowhttptest | 0.57 | 0.51 | 0.54 | 1,104 |

> [!NOTE]
> Minority attack classes (SSH-Patator, Bot, Web Attacks) have near-zero recall due to extreme class imbalance in CICIDS2017. The roadmap includes SMOTE/ADASYN oversampling and focal loss to address this.

### Anomaly Detection

| Model | Metric | Value | Status |
|---|---|---|:---:|
| **Autoencoder** | F1 @ optimal threshold | **0.86** | ✅ |

### RAG Pipeline (RAGAS)

| Metric | Score | What It Measures |
|---|---|---|
| **Faithfulness** | **0.82** | Are all claims in the answer inferable from retrieved CVE context? |
| **Context Precision** | Measured | Are the top retrieved chunks relevant to the question? |
| **Context Recall** | Measured | Are the ground-truth CVE references present in retrieved chunks? |
| **Answer Relevancy** | Measured | Does the answer directly address what was asked? |

### Fine-Tuning (LoRA)

| Model | Metric | Value |
|---|---|---|
| **LoRA DistilBERT** (threat classifier) | F1 macro | **0.79** |

### Unit & Integration Tests

| Module | Tests | Coverage |
|---|---|---|
| `tests/unit/test_pipeline.py` | sklearn pipeline build, fit, predict, save/load | ML pipeline |
| `tests/unit/test_autoencoder.py` | Autoencoder forward pass, fit, anomaly scoring, save/load | Anomaly detection |
| `tests/unit/test_agent_tools.py` | Tool invocation, graph compilation, ReAct routing | Agent + tools |
| `tests/integration/test_api.py` | Health endpoint, /predict, /agent/query, CORS, error handling | API integration |

```bash
pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data Engineering** | Apache Airflow, PySpark, DuckDB, S3 (Parquet) | ETL pipeline with local/cloud dual mode |
| **ML** | scikit-learn, XGBoost, LightGBM, SHAP, MLflow | Ensemble classification + explainability + experiment tracking |
| **Deep Learning** | PyTorch (Autoencoder), HuggingFace PEFT (LoRA) | Unsupervised anomaly detection + LLM fine-tuning |
| **RAG** | ChromaDB, `all-MiniLM-L6-v2`, LangChain retriever | CVE/threat intelligence vector search |
| **Agent** | LangGraph `StateGraph`, Groq/Together/Ollama | ReAct multi-tool agent with 3 free LLM backends |
| **Evaluation** | RAGAS, SHAP, AUC-PR, calibration curves | Automated ML + RAG quality benchmarks |
| **Serving** | FastAPI + Uvicorn, TorchServe, Docker (multi-stage) | Async REST API with Pydantic schemas |
| **Cloud** | AWS S3, EC2 (t3.medium), ECR, CloudWatch | Production infrastructure |
| **CI/CD** | GitHub Actions (5-stage pipeline) | Lint → Test → Build → Push ECR → Deploy EC2 |
| **Observability** | structlog, CloudWatch Logs + Metrics, drift detection | Structured logging + KL divergence drift alerting |

---

## Data

### Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset.

| Parameter | Specification |
|---|---|
| **Source** | [UNB CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) |
| **Total Records** | ~2.83M network flow records |
| **Raw Features** | 80 (packet lengths, flow duration, IAT, flags, byte counts, etc.) |
| **Engineered Features** | 11 (asymmetry ratios, flag densities, burst rates, IAT CV) |
| **Total Features** | 91 (80 base + 11 engineered) |
| **Attack Categories** | 15 classes: BENIGN, DoS Hulk, PortScan, DDoS, DoS GoldenEye, FTP-Patator, SSH-Patator, DoS slowloris, DoS Slowhttptest, Bot, Web Attack (Brute Force, XSS, SQL Injection), Infiltration, Heartbleed |
| **Train/Val Split** | 80/20 stratified random split |
| **Output Format** | Parquet (columnar, compressed) |

### RAG Corpus

| Parameter | Specification |
|---|---|
| **Source** | NVD REST API v2.0 + local threat intelligence JSON/text files |
| **Chunking** | 512 tokens (≈2048 chars), 64 token overlap |
| **Embedding** | `all-MiniLM-L6-v2` (384 dimensions, cosine similarity) |
| **Vector Store** | ChromaDB persistent collection (`threatmind_cves`) |
| **Metadata per chunk** | CVE ID, CVSS score, severity, CWE IDs, published date, attack type |

### CVE Corpus Architecture

| CWE Category | Example CWEs | Attack Relevance |
|---|---|---|
| Injection | CWE-89 (SQLi), CWE-77 (Command Injection), CWE-79 (XSS) | Web Attack classification grounding |
| Memory Safety | CWE-125 (OOB Read), CWE-787 (OOB Write), CWE-416 (Use-After-Free) | Buffer overflow / Heartbleed context |
| Resource Exhaustion | CWE-400 (DoS), CWE-918 (SSRF) | DoS/DDoS threat intelligence |
| Auth & Access | CWE-862/863 (Missing/Incorrect Authorization), CWE-352 (CSRF) | Brute force / infiltration context |
| Deserialization | CWE-502 | RCE vulnerability grounding (e.g., Log4Shell) |

---

## Where — Project Structure

```
threatmind/
│
├── airflow/
│   └── dags/
│       ├── ingest_cicids.py        # Airflow DAG: download → validate → PySpark clean → Parquet
│       └── train_pipeline.py       # Airflow DAG: load Parquet → train models → log to MLflow
│
├── src/
│   ├── preprocessing/
│   │   ├── spark_transforms.py     # PySpark/DuckDB: column rename, type cast, NaN/Inf handling, label encoding
│   │   └── feature_engineering.py  # 11 derived features: asymmetry ratios, flag densities, burst rates
│   │
│   ├── ml/
│   │   ├── pipeline.py             # sklearn Pipeline: Imputer → Scaler → VotingClassifier(XGB+LGB) + SHAP
│   │   ├── autoencoder.py          # PyTorch Autoencoder + AnomalyDetector wrapper (fit, score, threshold)
│   │   ├── evaluation.py           # AUC-PR, ROC-AUC, F1, classification report, PR curves
│   │   └── train.py                # MLflow entry point: trains both pipeline and autoencoder
│   │
│   ├── rag/
│   │   ├── ingest.py               # NVD API fetch + local corpus → chunk → embed → ChromaDB upsert
│   │   ├── retriever.py            # ThreatRAG: ChromaDB retrieval + LLM synthesis (Groq/Together/Ollama)
│   │   └── eval_ragas.py           # RAGAS evaluation: faithfulness, context recall, answer relevancy
│   │
│   ├── agent/
│   │   ├── graph.py                # LangGraph StateGraph: ReAct loop (agent ⇄ tools ⇄ END)
│   │   ├── tools.py                # 3 tools: ml_inference, rag_retrieval, nvd_lookup
│   │   └── prompts.py              # System prompt, few-shot examples, chain-of-thought template
│   │
│   ├── finetuning/
│   │   ├── lora_config.py          # LoRA hyperparameters + top-20 CWE label mapping
│   │   └── train_lora.py           # PEFT LoRA fine-tuning on DistilBERT for threat classification
│   │
│   └── serving/
│       ├── api.py                  # FastAPI: /predict, /agent/query, /health, /metrics + lifespan loader
│       ├── schemas.py              # Pydantic models for all request/response types
│       └── torchserve_handler.py   # TorchServe custom handler for PyTorch autoencoder
│
├── monitoring/
│   ├── cloudwatch_logger.py        # structlog → CloudWatch Logs + custom metrics (p50/p95/p99 latency)
│   └── drift_detector.py           # KL divergence on prediction score distribution vs. training baseline
│
├── tests/
│   ├── unit/
│   │   ├── test_pipeline.py        # Pipeline build, fit, predict, save/load
│   │   ├── test_autoencoder.py     # Autoencoder forward pass, anomaly scoring
│   │   └── test_agent_tools.py     # Tool invocation, graph compilation
│   └── integration/
│       └── test_api.py             # FastAPI endpoint integration tests
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml               # 5-stage CI/CD: lint → test → Docker build → ECR push → EC2 deploy
│
├── docker/
│   ├── Dockerfile.api              # Multi-stage Python 3.12 container (non-root, healthcheck)
│   └── Dockerfile.torchserve       # TorchServe container for PyTorch model serving
│
├── infra/
│   └── aws_setup.sh                # AWS bootstrap: S3 bucket + ECR repo + EC2 instance + security group
│
├── data/
│   ├── cve_corpus/                 # Local CVE/threat JSON files for RAG ingestion
│   └── processed/                  # train.parquet, val.parquet (gitignored)
│
├── models/                         # pipeline.joblib, autoencoder.pt, feature_names.json (gitignored)
├── chroma_db/                      # ChromaDB persistent vector store (gitignored)
├── reports/                        # classification_report.txt, anomaly_pr_curve.png, ragas_eval.json
│
├── docker-compose.yml              # FastAPI + MLflow + Ollama (optional) orchestration
├── pyproject.toml                  # ruff, black, pytest, coverage config
├── requirements.txt                # All Python dependencies (grouped by layer)
├── .env.example                    # Environment variable template (all free-tier)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker + Docker Compose
- AWS CLI configured (`aws configure`) — _optional for local development_
- One of the following LLM providers (all free):
  - **Groq API key** (recommended) — [console.groq.com](https://console.groq.com)
  - **Together AI API key** — [api.together.xyz](https://api.together.xyz)
  - **Ollama** installed locally — [ollama.com](https://ollama.com) (no API key needed)

### 1. Clone & Install

```bash
git clone https://github.com/RaajitSingh1306/ThreatMind.git
cd ThreatMind

python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env → set LLM_PROVIDER and corresponding API key
```

### 2. Data Pipeline (Local, DuckDB Mode)

```bash
# Download CICIDS2017 CSVs to archive/ (from UNB or Kaggle)
# Then run preprocessing:
python src/preprocessing/spark_transforms.py --input archive/ --output data/processed/
```

### 3. Train Models

```bash
python src/ml/train.py --experiment-name threatmind-v1
# Logs metrics + artifacts to MLflow. View at http://localhost:5000 after:
mlflow ui
```

### 4. Build RAG Index

```bash
# From NVD (fetches 2023+2024 CVEs by default):
python src/rag/ingest.py

# Or from local corpus:
python src/rag/ingest.py --corpus-dir data/cve_corpus/ --collection threatmind_cves
```

### 5. Run API Locally

```bash
uvicorn src.serving.api:app --reload --port 8000
```

```bash
# Health check
curl http://localhost:8000/health
# → {"status":"ok","pipeline_loaded":true,"autoencoder_loaded":true,"rag_available":true,"version":"0.1.0"}
```

### 6. Run Tests

```bash
pytest tests/ -v
```

### 7. Docker (Alternative)

```bash
docker compose up --build
# API:    http://localhost:8000
# MLflow: http://localhost:5000

# Optional: enable fully local LLM via Ollama
docker compose --profile ollama up
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — pipeline, autoencoder, and RAG status |
| `GET` | `/metrics` | Request metrics — total, predict, agent counts, avg latency, error rate |
| `POST` | `/predict` | ML classification + anomaly detection + SHAP explanations |
| `POST` | `/agent/query` | Natural language threat query routed through LangGraph ReAct agent |
| `GET` | `/docs` | Interactive Swagger/OpenAPI documentation |

### POST /predict — Request

```json
{
  "features": {
    "flow_duration": 0.002,
    "total_fwd_packets": 1.0,
    "total_bwd_packets": 0.0,
    "fwd_pkt_len_mean": 54.0,
    "syn_flag_count": 1.0
  }
}
```

### POST /predict — Response

```json
{
  "prediction": "DoS",
  "label_id": 1,
  "confidence": 0.94,
  "anomaly_score": 0.73,
  "is_anomaly": true,
  "shap_top_features": [
    {"feature": "total_fwd_packets", "value": 1.0, "shap": 0.42},
    {"feature": "flow_duration", "value": 0.002, "shap": 0.31},
    {"feature": "syn_flag_count", "value": 1.0, "shap": 0.28}
  ]
}
```

### POST /agent/query — Request

```json
{
  "query": "What CVEs are associated with Apache Log4j RCE and how severe are they?"
}
```

### POST /agent/query — Response

```json
{
  "answer": "Log4Shell (CVE-2021-44228) is a critical RCE in Apache Log4j with CVSS 10.0...",
  "sources": ["CVE-2021-44228", "CVE-2021-45046"],
  "tool_calls": ["rag_retrieval", "nvd_lookup"],
  "reasoning_trace": "Calling tools: ['rag_retrieval'] → Calling tools: ['nvd_lookup']"
}
```

---

## Deployment

### AWS Infrastructure Bootstrap

```bash
bash infra/aws_setup.sh
# Creates: S3 bucket (versioned, private), ECR repository (scan-on-push), EC2 instance (t3.medium)
# Outputs: connection details + GitHub Secrets to configure
```

| Resource | Purpose |
|---|---|
| **S3 bucket** | Raw data, Parquet, model artifacts, logs |
| **EC2** (t3.medium) | FastAPI + TorchServe + MLflow server |
| **ECR** | Docker image registry (scan-on-push enabled) |
| **CloudWatch** | Logs, custom metrics (latency, error rate, drift), alarms |

### CI/CD Pipeline

On push to `main`, the GitHub Actions pipeline ([`.github/workflows/ci_cd.yml`](.github/workflows/ci_cd.yml)) executes:

| Stage | What It Does |
|---|---|
| **1. Lint** | `ruff check` + `black --check` on `src/`, `tests/`, `monitoring/` |
| **2. Test** | `pytest tests/ -v` (unit tests, mocked LLM) |
| **3. Build** | `docker build` API image + smoke test (`curl /health`) |
| **4. Push ECR** | Tag + push to AWS ECR (skipped if AWS credentials not configured) |
| **5. Deploy EC2** | SSH → pull latest image → restart container → health check |

> [!NOTE]
> AWS credential steps are conditional — the CI pipeline still runs lint + test + build locally if `AWS_ACCESS_KEY_ID` is not set in GitHub Secrets.

---

## Monitoring

| Signal | Method | Target |
|---|---|---|
| **Prediction distribution** | KL divergence vs. training baseline (1-hour windows) | Alert if KL > 0.1 |
| **Request latency** | p50 / p95 / p99 via `RequestMetricsCollector` | CloudWatch custom metric |
| **Error rate** | Tracked per request via middleware | Alert if > 1% over 5 min |
| **Drift detection** | Symmetric KL divergence on max-class probability histogram | Publishes `DriftAlert` metric to CloudWatch |

Structured logs are emitted via `structlog` and shipped to CloudWatch Logs (when AWS credentials are configured). In local mode, logs stream to stdout as human-readable JSON.

---

## Fine-Tuning (LoRA)

DistilBERT fine-tuned on threat category classification using NVD CWE descriptions as labels:

| Parameter | Value |
|---|---|
| **Base model** | `distilbert-base-uncased` |
| **PEFT method** | LoRA |
| **Rank** | 8 |
| **Alpha** | 32 |
| **Dropout** | 0.1 |
| **Target modules** | `q_lin`, `v_lin` |
| **Num labels** | 20 (top CWE categories: XSS, SQLi, Buffer Errors, DoS, SSRF, etc.) |
| **Epochs** | 3 |
| **Learning rate** | 2e-4 |
| **Max sequence length** | 256 |

```bash
python src/finetuning/train_lora.py --epochs 3 --output models/lora_threat_clf
```

---

## RAG Evaluation

```bash
python src/rag/eval_ragas.py --testset data/rag_testset.json
# Outputs: reports/ragas_eval.json
# Metrics: faithfulness, context_precision, context_recall, answer_relevancy
```

Falls back to a built-in synthetic test set (Log4Shell, SYN flood, Heartbleed) if no testset file is provided.

---

## Connected Projects

| Project | Role | Repository |
|---|---|---|
| **ThreatMind** (This Repo) | Cybersecurity ML + LLM agent platform | [ThreatMind](https://github.com/RaajitSingh1306/ThreatMind) |
| **SEBI RAG Bot** | Multi-agent compliance Q&A for Indian financial regulations | [sebi-rag-bot](https://github.com/RaajitSingh1306/sebi-rag-bot) |
| **Volatility Intelligence Platform** | Production GARCH + HMM + XGBoost market intelligence API | [volatility-intelligence-platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform) |
| **Credit Default Predictor** | Loan default prediction & TreeSHAP explainability engine | [Credit-Default-Predictor](https://github.com/RaajitSingh1306/Credit-Default-Predictor) |
| **NSEI Daily Stock Pipeline** | Financial data lakehouse & feature store (Airflow, Spark, DuckDB) | [NSEI-Daily-Stock-Pipeline](https://github.com/RaajitSingh1306/NSEI-Daily-Stock-Pipeline) |

---

## Limitations & Roadmap

### Known Limitations

- **Class Imbalance**: CICIDS2017 is heavily skewed — BENIGN accounts for ~80% of records. Minority attack classes (SSH-Patator, Bot, Web Attacks, Infiltration, Heartbleed) have near-zero recall with the current unweighted training.
- **Single Dense Retrieval**: The RAG pipeline uses dense-only vector search via ChromaDB. No sparse (BM25) component means exact CVE ID searches (`CVE-2021-44228`) rely entirely on embedding similarity, which can miss lexically specific queries.
- **In-Memory BM25 Absent**: Unlike the SEBI RAG Bot's hybrid retrieval, ThreatMind lacks a BM25 sparse retrieval layer for exact-term matching on CVE IDs and CWE codes.
- **Fixed Autoencoder Threshold**: The 95th percentile threshold is computed once at training time and not updated online as the production data distribution evolves.
- **NVD Rate Limiting**: Without an API key, NVD fetches are limited to 5 requests per 30 seconds, making large corpus builds slow (~6 hours for a single year).
- **No Multi-Turn Memory**: The LangGraph agent processes each query independently — no session context or coreference resolution across conversation turns.
- **Groq Free-Tier Limits**: Production inference is subject to Groq's rate limits (~30 requests/minute), requiring client-side throttling for burst traffic.

### Roadmap

- [ ] **Class Imbalance Mitigation**: Integrate SMOTE/ADASYN oversampling for minority classes and focal loss in XGBoost to improve recall on rare attack types.
- [ ] **Hybrid Retrieval**: Add BM25 sparse retrieval (Okapi) alongside ChromaDB dense search with weighted fusion scoring for CVE ID exact matching.
- [ ] **Cross-Encoder Reranking**: Deploy a cross-encoder reranker (`bge-reranker-large`) over top-20 candidates to optimise top-5 precision.
- [ ] **Online Threshold Adaptation**: Implement sliding-window autoencoder threshold recalibration using recent benign traffic samples.
- [ ] **MITRE ATT&CK Mapping**: Extend the RAG corpus and agent tools to include MITRE ATT&CK technique/tactic mappings alongside CVE data.
- [ ] **Session & Thread Memory**: Add Redis/PostgreSQL-backed conversational memory for multi-turn threat analysis sessions.
- [ ] **Streaming Inference**: Replace batch prediction with streaming network flow analysis for real-time IDS deployment.
- [ ] **Terraform IaC**: Replace `aws_setup.sh` with Terraform modules for reproducible infrastructure provisioning.

---

## License

MIT License. Built for cybersecurity threat intelligence research and production ML engineering demonstration.
