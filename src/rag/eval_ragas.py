"""
RAGAS evaluation for ThreatMind RAG pipeline.

Evaluates: faithfulness, context_precision, context_recall, answer_relevancy.

Usage:
    python src/rag/eval_ragas.py --testset data/rag_testset.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

# Synthetic test set for offline eval (used when no testset file is provided)
SAMPLE_TESTSET = [
    {
        "question": "What is Log4Shell and which CVEs are associated with it?",
        "ground_truth": "Log4Shell is a critical RCE vulnerability in Apache Log4j. CVE-2021-44228 is the primary CVE with CVSS 10.0.",
    },
    {
        "question": "What CVEs are related to SYN flood DoS attacks?",
        "ground_truth": "SYN flood is a form of DoS that exploits TCP handshake. Various CVEs in network stacks relate to improper SYN handling.",
    },
    {
        "question": "How severe is Heartbleed (CVE-2014-0160)?",
        "ground_truth": "CVE-2014-0160 (Heartbleed) is a critical buffer over-read in OpenSSL with CVSS 7.5 that exposes private memory.",
    },
]


def run_evaluation(testset_path: str | None, output_path: str = "reports/ragas_eval.json") -> dict:
    """
    Run RAGAS evaluation on the RAG pipeline.
    Falls back to synthetic test set if no testset file provided.
    """
    from src.rag.retriever import ThreatRAG  # noqa: PLC0415

    rag = ThreatRAG()

    if testset_path and Path(testset_path).exists():
        with open(testset_path) as f:
            testset = json.load(f)
        log.info("Loaded testset", n=len(testset), path=testset_path)
    else:
        log.warning("No testset provided — using synthetic sample set", n=len(SAMPLE_TESTSET))
        testset = SAMPLE_TESTSET

    # Collect RAG outputs
    questions, answers, contexts, ground_truths = [], [], [], []
    for item in testset:
        q = item["question"]
        gt = item.get("ground_truth", "")
        result = rag.answer(q)
        retrieved = rag.retrieve(q)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in retrieved])
        ground_truths.append(gt)
        log.info("Evaluated", question=q[:60])

    # RAGAS evaluation
    try:
        from datasets import Dataset  # noqa: PLC0415
        from ragas import evaluate  # noqa: PLC0415
        from ragas.metrics import (  # noqa: PLC0415
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(dataset, metrics=[faithfulness, context_precision, context_recall, answer_relevancy])
        scores = {
            "faithfulness": float(result["faithfulness"]),
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
            "answer_relevancy": float(result["answer_relevancy"]),
        }
    except Exception as e:
        log.warning("RAGAS evaluation failed, computing simple metrics", error=str(e))
        # Simple fallback: answer length as proxy
        scores = {
            "faithfulness": None,
            "context_precision": None,
            "context_recall": None,
            "answer_relevancy": None,
            "error": str(e),
            "n_evaluated": len(questions),
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    log.info("RAGAS evaluation complete", scores=scores, output=output_path)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="ThreatMind RAGAS evaluation")
    parser.add_argument("--testset", default=None, help="Path to testset JSON")
    parser.add_argument("--output", default="reports/ragas_eval.json")
    args = parser.parse_args()
    run_evaluation(args.testset, args.output)


if __name__ == "__main__":
    main()
