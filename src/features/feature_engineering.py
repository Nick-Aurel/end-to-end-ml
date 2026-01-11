"""Feature engineering utilities."""

from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features using StandardScaler.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
