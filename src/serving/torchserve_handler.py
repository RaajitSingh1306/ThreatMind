"""
TorchServe handler for ThreatMind Autoencoder.

Used when deploying the autoencoder via TorchServe for native PyTorch serving.
Handles serialisation/deserialisation and returns anomaly scores.
"""

from __future__ import annotations

import json
import os

import torch
from ts.torch_handler.base_handler import BaseHandler


class AutoencoderHandler(BaseHandler):
    """TorchServe handler for the ThreatMind anomaly detection autoencoder."""

    def __init__(self):
        super().__init__()
        self.threshold: float | None = None
        self.initialized = False

    def initialize(self, context) -> None:
        """Load model from TorchServe model store."""
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        manifest = context.manifest
        model_file = manifest["model"].get("modelFile", "autoencoder.pt")
        model_path = os.path.join(model_dir, model_file)

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

        from src.ml.autoencoder import Autoencoder  # noqa: PLC0415

        self.model = Autoencoder(input_dim=checkpoint["input_dim"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.threshold = checkpoint.get("threshold", 0.01)

        self.initialized = True

    def preprocess(self, data: list[dict]) -> torch.Tensor:
        """Convert request data to tensor."""
        features_list = []
        for row in data:
            body = row.get("body") or row.get("data", {})
            if isinstance(body, (bytes, bytearray)):
                body = json.loads(body.decode("utf-8"))
            features = body.get("features", body)
            if isinstance(features, dict):
                features = list(features.values())
            features_list.append(features)
        return torch.FloatTensor(features_list)

    def inference(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            recon = self.model(inputs)
            errors = torch.mean((recon - inputs) ** 2, dim=1)
        scores = 1.0 / (1.0 + torch.exp(-(errors / self.threshold - 1.0))) if self.threshold else errors
        return errors, scores

    def postprocess(self, outputs: tuple[torch.Tensor, torch.Tensor]) -> list[dict]:
        errors, scores = outputs
        results = []
        for err, score in zip(errors.numpy(), scores.numpy(), strict=False):
            results.append({
                "reconstruction_error": float(err),
                "anomaly_score": float(score),
                "is_anomaly": float(err) > (self.threshold or 0.01),
            })
        return results
