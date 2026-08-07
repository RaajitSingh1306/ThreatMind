"""
LoRA configuration for DistilBERT threat classification fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning via PEFT."""

    # Base model
    base_model_name: str = "distilbert-base-uncased"

    # LoRA hyperparameters
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q_lin", "v_lin")
    bias: str = "none"
    task_type: str = "SEQ_CLS"

    # Training
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 256

    # Output
    output_dir: str = "models/lora_threat_clf"

    # Data
    # Labels: CWE categories derived from NVD
    num_labels: int = 20  # top-20 CWE categories

    def to_peft_config(self):
        """Return a PEFT LoraConfig object."""
        from peft import LoraConfig, TaskType  # noqa: PLC0415

        return LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=list(self.target_modules),
            bias=self.bias,
            task_type=TaskType.SEQ_CLS,
        )


# Default config instance
DEFAULT_LORA_CONFIG = LoRAConfig()

# Top CWE categories used as classification labels
CWE_LABELS = [
    "CWE-79",    # XSS
    "CWE-89",    # SQL Injection
    "CWE-20",    # Improper Input Validation
    "CWE-125",   # Out-of-bounds Read
    "CWE-787",   # Out-of-bounds Write
    "CWE-119",   # Buffer Errors
    "CWE-200",   # Information Exposure
    "CWE-416",   # Use After Free
    "CWE-22",    # Path Traversal
    "CWE-352",   # CSRF
    "CWE-77",    # Command Injection
    "CWE-190",   # Integer Overflow
    "CWE-400",   # Resource Exhaustion (DoS)
    "CWE-502",   # Deserialization
    "CWE-611",   # XXE
    "CWE-732",   # Incorrect Permission Assignment
    "CWE-862",   # Missing Authorization
    "CWE-863",   # Incorrect Authorization
    "CWE-918",   # SSRF
    "OTHER",     # Catch-all
]

CWE_TO_ID = {cwe: i for i, cwe in enumerate(CWE_LABELS)}
ID_TO_CWE = dict(enumerate(CWE_LABELS))
