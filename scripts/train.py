#!/usr/bin/env python3
"""CLI script for training models."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train import train_model


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train a machine learning model")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset for testing (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize features before training",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/trained_model.joblib",
        help="Path to save the trained model (default: models/trained_model.joblib)",
    )

    args = parser.parse_args()

    train_model(
        test_size=args.test_size,
        random_state=args.random_state,
        standardize=args.standardize,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
