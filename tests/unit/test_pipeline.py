"""
Unit tests for the sklearn ML pipeline (XGBoost/LGB).
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification


def _make_synthetic_data(n_samples=200, n_classes=5):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X.astype(np.float32), y


class TestBuildPipeline:
    def test_pipeline_builds(self):
        from src.ml.pipeline import build_pipeline

        pipeline = build_pipeline(n_classes=5)
        assert pipeline is not None
        assert "imputer" in pipeline.named_steps
        assert "scaler" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps

    def test_pipeline_fit_predict(self):
        from src.ml.pipeline import build_pipeline

        X, y = _make_synthetic_data(n_classes=5)
        pipeline = build_pipeline(n_classes=5)
        pipeline.fit(X, y)

        preds = pipeline.predict(X[:10])
        assert len(preds) == 10
        assert all(0 <= p < 5 for p in preds)

    def test_pipeline_predict_proba(self):
        from src.ml.pipeline import build_pipeline

        X, y = _make_synthetic_data(n_classes=5)
        pipeline = build_pipeline(n_classes=5)
        pipeline.fit(X, y)

        proba = pipeline.predict_proba(X[:5])
        assert proba.shape == (5, 5)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    def test_pipeline_handles_nan(self):
        """Pipeline should handle NaN inputs via imputer."""
        from src.ml.pipeline import build_pipeline

        X, y = _make_synthetic_data(n_classes=3)
        X_with_nan = X.copy()
        X_with_nan[0, 0] = np.nan
        X_with_nan[1, 3] = np.nan

        pipeline = build_pipeline(n_classes=3)
        pipeline.fit(X_with_nan, y)
        preds = pipeline.predict(X_with_nan[:5])
        assert len(preds) == 5


class TestSHAP:
    def test_shap_returns_features(self):
        from src.ml.pipeline import build_pipeline, get_shap_top_features

        X, y = _make_synthetic_data(n_classes=3)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        pipeline = build_pipeline(n_classes=3)
        pipeline.fit(X, y)

        result = get_shap_top_features(pipeline, X[:1], feature_names, top_n=5)
        assert len(result) == 5
        for item in result:
            assert "feature" in item
            assert "value" in item
            assert "shap" in item
