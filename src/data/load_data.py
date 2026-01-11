"""Data loading utilities for the ML project."""

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.datasets import load_iris


def load_iris_data() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load the Iris dataset.

    Returns:
        Tuple of (features, target, target_names)
    """
    data = load_iris()
    return data.data, data.target, data.target_names.tolist()


def save_processed_data(X: np.ndarray, y: np.ndarray, filename: str = "processed_data.npy") -> None:
    """
    Save processed data to disk.

    Args:
        X: Feature matrix
        y: Target vector
        filename: Name of the file to save
    """
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = data_dir / filename
    np.savez(filepath, X=X, y=y)
    print(f"Processed data saved to {filepath}")


def load_processed_data(filename: str = "processed_data.npy") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load processed data from disk.

    Args:
        filename: Name of the file to load

    Returns:
        Tuple of (features, target)
    """
    filepath = Path("data/processed") / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data file not found: {filepath}")
    
    data = np.load(filepath, allow_pickle=True)
    return data["X"], data["y"]
