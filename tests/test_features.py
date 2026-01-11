"""Tests for feature engineering modules."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.features import standardize_features


def test_standardize_features():
    """Test feature standardization."""
    # Create dummy data
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_test = np.array([[2.0, 3.0], [4.0, 5.0]])

    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)

    # Check types
    assert isinstance(scaler, StandardScaler)
    assert isinstance(X_train_scaled, np.ndarray)
    assert isinstance(X_test_scaled, np.ndarray)

    # Check shapes
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape

    # Check that training data is standardized (mean ~0, std ~1)
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-10)
