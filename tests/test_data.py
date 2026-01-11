"""Tests for data loading modules."""

import numpy as np
import pytest

from src.data import load_iris_data


def test_load_iris_data():
    """Test loading Iris dataset."""
    X, y, target_names = load_iris_data()

    # Check shapes
    assert X.shape[0] == 150  # 150 samples
    assert X.shape[1] == 4  # 4 features
    assert len(y) == 150  # 150 labels

    # Check target names
    assert len(target_names) == 3
    assert isinstance(target_names, list)

    # Check data types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
