"""Tests for model training and prediction."""

import pytest
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from src.models.train import train_model
from src.models.predict import predict


def test_train_model(tmp_path):
    """Test model training."""
    model_path = str(tmp_path / "test_model.joblib")

    model = train_model(
        test_size=0.2,
        random_state=42,
        standardize=False,
        model_path=model_path,
    )

    # Check model type
    assert isinstance(model, LogisticRegression)

    # Check model file exists
    assert Path(model_path).exists()


def test_predict_with_model(tmp_path):
    """Test prediction functionality."""
    # First train a model
    model_path = str(tmp_path / "test_model.joblib")
    train_model(
        test_size=0.2,
        random_state=42,
        standardize=False,
        model_path=model_path,
    )

    # Test prediction with sample features
    features = [5.1, 3.5, 1.4, 0.2]
    predict(model_path=model_path, features=features, num_samples=1)
