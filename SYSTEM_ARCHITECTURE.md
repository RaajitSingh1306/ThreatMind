# ThreatMind — System Architecture

## Overview

ThreatMind is a cybersecurity threat intelligence platform combining classical ML (intrusion detection), deep learning (anomaly detection), a RAG pipeline over CVE/threat corpora, and a LangGraph agentic layer — all deployed on AWS with full CI/CD and observability.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                  │
│                                                                      │
│   CICIDS2017 CSVs ──► Airflow DAG ──► PySpark ──► Parquet (S3)     │
│   NVD CVE JSON  ──────────────────────────────────► ChromaDB        │
│   Threat Reports ─────────────────────────────────► ChromaDB        │
│                              │                                       │
│                           DuckDB (local dev)                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                          ML LAYER                                    │
│                                                                      │
│   sklearn Pipeline                    PyTorch Autoencoder           │
│   ├─ Imputation                       ├─ Encoder (FC layers)        │
│   ├─ Scaling                          ├─ Decoder (FC layers)        │
│   ├─ Feature Engineering             ├─ Reconstruction error thresh  │
│   ├─ XGBoost Classifier              └─ MLflow artifact             │
│   ├─ LightGBM (ensemble)                                            │
│   └─ SHAP explainability                                            │
│                                                                      │
│   MLflow Tracking Server (EC2) ◄──── all experiments               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        SERVING LAYER                                 │
│                                                                      │
│   FastAPI (main gateway)              TorchServe                    │
│   ├─ POST /predict          ◄────────► PyTorch autoencoder          │
│   ├─ POST /agent/query                                              │
│   ├─ GET  /health                                                   │
│   └─ GET  /metrics                                                  │
│                                                                      │
│   Docker container ──► ECR ──► EC2                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                     RAG + AGENT LAYER                                │
│                                                                      │
│   LangGraph Agent (ReAct loop)                                      │
│   ├─ Tool 1: ML Inference  ──► FastAPI /predict                     │
│   ├─ Tool 2: RAG Retrieval ──► ChromaDB → LLM answer               │
│   └─ Tool 3: CVE Lookup    ──► NVD REST API                        │
│                                                                      │
│   RAG Pipeline                                                      │
│   ├─ Corpus: NVD CVE JSON + threat PDFs                             │
│   ├─ Chunking: 512 tokens, 64 overlap                               │
│   ├─ Embedding: all-MiniLM-L6-v2 (HuggingFace)                     │
│   ├─ Vector Store: ChromaDB (persistent)                            │
│   └─ LLM: OpenAI GPT-4o / Anthropic Claude API                     │
│                                                                      │
│   Evaluation: RAGAS (faithfulness, context recall, answer relevance) │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      FINE-TUNING MODULE                              │
│                                                                      │
│   Base: DistilBERT (HuggingFace)                                    │
│   Task: Threat category classification (NVD CWE labels)             │
│   Method: LoRA via PEFT library                                     │
│   ├─ rank=8, alpha=32, target_modules=["q_lin","v_lin"]             │
│   └─ Trainer: HuggingFace Trainer + Seq2SeqTrainer                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                             │
│                                                                      │
│   AWS                                                               │
│   ├─ S3:   raw data, Parquet, model artifacts, logs                 │
│   ├─ EC2:  FastAPI app, TorchServe, MLflow server                   │
│   ├─ ECR:  Docker image registry                                    │
│   └─ CloudWatch: logs, metrics, alerts                              │
│                                                                      │
│   CI/CD: GitHub Actions                                             │
│   ├─ lint (ruff, black)                                             │
│   ├─ pytest (unit + integration)                                    │
│   ├─ docker build + push ECR                                        │
│   └─ SSH deploy to EC2                                              │
│                                                                      │
│   Monitoring                                                        │
│   ├─ structlog → CloudWatch Logs                                    │
│   ├─ prediction distribution (drift proxy)                          │
│   ├─ request latency (p50/p95/p99)                                  │
│   └─ error rate alerting                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow — Inference Request

```
User Query
    │
    ▼
FastAPI /agent/query
    │
    ▼
LangGraph Agent (ReAct loop)
    ├── Decide: needs ML prediction?
    │       └──► POST FastAPI /predict ──► XGBoost / TorchServe
    │                   └──► SHAP explanation returned
    │
    ├── Decide: needs threat context?
    │       └──► RAG Tool
    │               ├─ embed query (MiniLM)
    │               ├─ ChromaDB similarity search (top-k=5)
    │               └─ LLM synthesize answer
    │
    └── Decide: needs CVE data?
            └──► NVD API /cves/2.0?keywordSearch=...
                        └─ structured CVE metadata returned
    │
    ▼
LLM final synthesis (GPT-4o / Claude)
    │
    ▼
Structured JSON response → client
```

---

## Data Flow — Training Pipeline

```
S3 (raw CSVs)
    │
    ▼
Airflow DAG: ingest_cicids
    ├─ Task 1: download + validate
    ├─ Task 2: PySpark clean (nulls, dtypes, label encoding)
    ├─ Task 3: PySpark feature engineering
    └─ Task 4: write Parquet → S3 (processed/)
    │
    ▼
Train DAG (triggered after ingest)
    ├─ Task 1: load Parquet → pandas
    ├─ Task 2: sklearn pipeline fit (XGBoost + LGB)
    ├─ Task 3: PyTorch autoencoder train loop
    ├─ Task 4: MLflow log metrics + artifacts
    └─ Task 5: push model to S3 (models/)
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Vector DB | ChromaDB | Persistent, local-first, no infra overhead vs Pinecone |
| Embedding model | all-MiniLM-L6-v2 | Fast, free, sufficient for CVE domain |
| Agent framework | LangGraph | Stateful, explicit graph — better than LangChain for multi-tool |
| Anomaly detection | PyTorch Autoencoder | Unsupervised — no labels needed for novel attack types |
| ML serving split | FastAPI + TorchServe | XGBoost via FastAPI (simple), PyTorch via TorchServe (native) |
| Deployment | EC2 (not Lambda) | Model loading latency makes Lambda cold starts unacceptable |
| Fine-tuning | LoRA via PEFT | Parameter efficiency — full fine-tune unnecessary at this scale |
| Drift detection | Prediction distribution | Simple, interpretable proxy without ground truth labels in prod |

---

## Directory Structure

```
threatmind/
├── airflow/
│   └── dags/
│       ├── ingest_cicids.py
│       └── train_pipeline.py
├── data/
│   └── raw/                    # gitignored, pulled from S3
├── src/
│   ├── preprocessing/
│   │   ├── spark_transforms.py
│   │   └── feature_engineering.py
│   ├── ml/
│   │   ├── pipeline.py         # sklearn pipeline + XGBoost/LGB
│   │   ├── autoencoder.py      # PyTorch model + training loop
│   │   ├── evaluation.py       # AUC, PR, calibration, SHAP
│   │   └── train.py            # MLflow experiment entry point
│   ├── rag/
│   │   ├── ingest.py           # chunk, embed, store to ChromaDB
│   │   ├── retriever.py        # query pipeline
│   │   └── eval_ragas.py       # RAGAS evaluation suite
│   ├── agent/
│   │   ├── graph.py            # LangGraph state machine
│   │   ├── tools.py            # 3 tools: ML, RAG, NVD
│   │   └── prompts.py          # ReAct, CoT, few-shot templates
│   ├── finetuning/
│   │   ├── lora_config.py
│   │   └── train_lora.py
│   └── serving/
│       ├── api.py              # FastAPI app
│       ├── torchserve_handler.py
│       └── schemas.py          # Pydantic models
├── monitoring/
│   ├── cloudwatch_logger.py
│   └── drift_detector.py
├── tests/
│   ├── unit/
│   └── integration/
├── .github/
│   └── workflows/
│       └── ci_cd.yml
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.torchserve
├── infra/
│   └── aws_setup.sh            # EC2, ECR, S3 bootstrap
├── notebooks/
│   └── eda_cicids.ipynb
├── SYSTEM_ARCHITECTURE.md
└── README.md
```

---

## ML Metrics Targets

| Model | Primary Metric | Target |
|---|---|---|
| XGBoost classifier | AUC-PR | > 0.92 |
| LightGBM classifier | ROC-AUC | > 0.97 |
| Autoencoder (anomaly) | F1 @ threshold | > 0.85 |
| RAG pipeline | RAGAS faithfulness | > 0.80 |
| LoRA classifier | F1 macro | > 0.78 |
