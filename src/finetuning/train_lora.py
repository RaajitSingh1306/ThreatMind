"""
LoRA fine-tuning script for ThreatMind threat category classifier.

Fine-tunes DistilBERT on CVE description → CWE category classification.
Training data is synthesised from NVD CVE data or loaded from disk.

Usage:
    python src/finetuning/train_lora.py --epochs 3 --output models/lora_threat_clf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _build_dataset_from_nvd(years: list[int] = None, max_per_year: int = 5000):
    """
    Build classification dataset from NVD CVE descriptions.
    Maps CVE description → primary CWE category.
    """
    from src.finetuning.lora_config import CWE_TO_ID  # noqa: PLC0415
    from src.rag.ingest import fetch_nvd_cves  # noqa: PLC0415

    years = years or [2023, 2024]
    texts, labels = [], []

    api_key = os.getenv("NVD_API_KEY") or None

    for year in years:
        log.info("Fetching CVEs for fine-tuning dataset", year=year)
        cves = fetch_nvd_cves(year, api_key)
        for v in cves[:max_per_year]:
            cve = v.get("cve", {})
            desc = next(
                (d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"),
                "",
            )
            if len(desc) < 30:
                continue

            # Extract primary CWE
            weaknesses = cve.get("weaknesses", [])
            cwe = "OTHER"
            for w in weaknesses:
                for d in w.get("description", []):
                    candidate = d.get("value", "")
                    if candidate in CWE_TO_ID:
                        cwe = candidate
                        break
                if cwe != "OTHER":
                    break

            texts.append(desc[:512])
            labels.append(CWE_TO_ID[cwe])

    log.info("Dataset built", n=len(texts))
    return texts, labels


def train(
    output_dir: str = "models/lora_threat_clf",
    epochs: int = 3,
    data_path: str | None = None,
) -> None:
    """Fine-tune DistilBERT with LoRA for CWE classification."""
    try:
        from datasets import Dataset  # noqa: PLC0415
        from peft import get_peft_model  # noqa: PLC0415
        from sklearn.model_selection import train_test_split  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        log.error("Missing dependency for fine-tuning", error=str(e))
        sys.exit(1)

    from src.finetuning.lora_config import (  # noqa: PLC0415
        CWE_LABELS,
        DEFAULT_LORA_CONFIG,
        ID_TO_CWE,
    )

    cfg = DEFAULT_LORA_CONFIG
    cfg.num_epochs = epochs
    cfg.output_dir = output_dir

    # ── Dataset ───────────────────────────────────────────────────────────────
    if data_path and Path(data_path).exists():
        log.info("Loading dataset from file", path=data_path)
        with open(data_path) as f:
            data = json.load(f)
        texts = [d["text"] for d in data]
        labels = [d["label"] for d in data]
    else:
        log.info("Building dataset from NVD API")
        texts, labels = _build_dataset_from_nvd(years=[2023, 2024], max_per_year=3000)

    if not texts:
        log.error("No training data available")
        sys.exit(1)

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.15, random_state=42)

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_seq_length, padding="max_length")

    train_ds = Dataset.from_dict({"text": X_train, "label": y_train}).map(tokenize, batched=True)
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val}).map(tokenize, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # ── Model ─────────────────────────────────────────────────────────────────
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model_name,
        num_labels=len(CWE_LABELS),
        id2label=ID_TO_CWE,
        label2id={v: k for k, v in ID_TO_CWE.items()},
    )
    model = get_peft_model(base_model, cfg.to_peft_config())
    model.print_trainable_parameters()

    # ── Training ──────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "threatmind-v1"))

    with mlflow.start_run(run_name="lora_distilbert"):
        mlflow.log_params({
            "base_model": cfg.base_model_name,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "target_modules": cfg.target_modules,
            "epochs": epochs,
            "n_train": len(X_train),
            "n_val": len(X_val),
        })

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=50,
            report_to="none",  # use MLflow manually
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        mlflow.log_metrics({"eval_loss": eval_results.get("eval_loss", 0)})

        # Save model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        mlflow.log_artifact(output_dir)
        log.info("LoRA fine-tuning complete", output_dir=output_dir, eval=eval_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="ThreatMind LoRA fine-tuning")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default="models/lora_threat_clf")
    parser.add_argument("--data", default=None, help="Optional JSON dataset file")
    args = parser.parse_args()
    train(output_dir=args.output, epochs=args.epochs, data_path=args.data)


if __name__ == "__main__":
    main()
