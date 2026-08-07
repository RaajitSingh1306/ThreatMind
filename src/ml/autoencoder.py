"""
PyTorch Autoencoder for unsupervised anomaly detection on CICIDS2017 features.

Architecture:
    Input (N features) → FC(256) → ReLU → FC(128) → ReLU → FC(64) [bottleneck]
    → FC(128) → ReLU → FC(256) → ReLU → FC(N) [reconstruction]

Anomaly score = mean squared reconstruction error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

log = structlog.get_logger()


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AnomalyDetector:
    """
    Wraps Autoencoder with threshold-based anomaly detection.
    Train on BENIGN-only data, then threshold on reconstruction error.
    """

    def __init__(self, input_dim: int, latent_dim: int = 64, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder(input_dim, latent_dim).to(self.device)
        self.threshold: float | None = None
        self.input_dim = input_dim

    def fit(
        self,
        X_benign: np.ndarray,
        epochs: int = 30,
        batch_size: int = 512,
        lr: float = 1e-3,
        threshold_percentile: float = 95.0,
    ) -> list[float]:
        """Train on benign-only data and set reconstruction-error threshold."""
        X_t = torch.FloatTensor(X_benign).to(self.device)
        dataset = TensorDataset(X_t, X_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        epoch_losses: list[float] = []
        self.model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for X_batch, _ in loader:
                optimizer.zero_grad()
                recon = self.model(X_batch)
                loss = criterion(recon, X_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(X_batch)
            epoch_loss = running_loss / len(X_benign)
            epoch_losses.append(epoch_loss)
            scheduler.step()
            if epoch % 5 == 0:
                log.info("Autoencoder training", epoch=epoch, loss=round(epoch_loss, 6))

        # Compute threshold on training data
        self.threshold = float(
            np.percentile(self.reconstruction_errors(X_benign), threshold_percentile)
        )
        log.info("Threshold set", percentile=threshold_percentile, threshold=self.threshold)
        return epoch_losses

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample MSE reconstruction errors."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            recon = self.model(X_t)
            errors = torch.mean((recon - X_t) ** 2, dim=1).cpu().numpy()
        return errors

    def predict_anomaly(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (anomaly_flags, reconstruction_errors).
        anomaly_flag = 1 if error > threshold.
        """
        if self.threshold is None:
            raise RuntimeError("Call fit() before predict_anomaly()")
        errors = self.reconstruction_errors(X)
        flags = (errors > self.threshold).astype(int)
        return flags, errors

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Normalised anomaly score in [0, 1] based on sigmoid of error/threshold."""
        errors = self.reconstruction_errors(X)
        if self.threshold and self.threshold > 0:
            return 1.0 / (1.0 + np.exp(-(errors / self.threshold - 1.0)))
        return errors

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "threshold": self.threshold,
                "input_dim": self.input_dim,
            },
            path,
        )
        log.info("Autoencoder saved", path=path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> AnomalyDetector:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        detector = cls(input_dim=checkpoint["input_dim"], device=device)
        detector.model.load_state_dict(checkpoint["model_state"])
        detector.threshold = checkpoint["threshold"]
        log.info("Autoencoder loaded", path=path, threshold=detector.threshold)
        return detector
