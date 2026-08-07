"""
Unit tests for PyTorch Autoencoder.
"""

import numpy as np
import pytest
import torch


class TestAutoencoder:
    def test_forward_pass(self):
        from src.ml.autoencoder import Autoencoder

        model = Autoencoder(input_dim=50)
        x = torch.randn(8, 50)
        out = model(x)
        assert out.shape == (8, 50)

    def test_reconstruction_shape(self):
        from src.ml.autoencoder import AnomalyDetector

        detector = AnomalyDetector(input_dim=30)
        X = np.random.randn(100, 30).astype(np.float32)
        # Only test forward without training
        errors = detector.reconstruction_errors(X)
        assert errors.shape == (100,)
        assert (errors >= 0).all()

    def test_fit_sets_threshold(self):
        from src.ml.autoencoder import AnomalyDetector

        detector = AnomalyDetector(input_dim=20)
        X_benign = np.random.randn(200, 20).astype(np.float32)
        detector.fit(X_benign, epochs=2, batch_size=64)
        assert detector.threshold is not None
        assert detector.threshold > 0

    def test_predict_anomaly(self):
        from src.ml.autoencoder import AnomalyDetector

        detector = AnomalyDetector(input_dim=20)
        X_benign = np.random.randn(200, 20).astype(np.float32) * 0.1
        detector.fit(X_benign, epochs=2, batch_size=64)

        # Anomalous sample: very different distribution
        X_anomaly = np.random.randn(10, 20).astype(np.float32) * 100
        flags, errors = detector.predict_anomaly(X_anomaly)
        assert flags.shape == (10,)
        assert errors.shape == (10,)

    def test_save_load(self, tmp_path):
        from src.ml.autoencoder import AnomalyDetector

        detector = AnomalyDetector(input_dim=20)
        X = np.random.randn(100, 20).astype(np.float32)
        detector.fit(X, epochs=2, batch_size=32)

        save_path = str(tmp_path / "ae.pt")
        detector.save(save_path)

        loaded = AnomalyDetector.load(save_path)
        assert loaded.threshold == detector.threshold
        assert loaded.input_dim == detector.input_dim

        # Predictions should match
        e1 = detector.reconstruction_errors(X[:5])
        e2 = loaded.reconstruction_errors(X[:5])
        np.testing.assert_allclose(e1, e2, rtol=1e-4)
