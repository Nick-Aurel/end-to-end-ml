#!/usr/bin/env python3
"""CLI script for making predictions."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predict import predict


def main():
    """CLI entry point for predictions."""
    parser = argparse.ArgumentParser(description="Make predictions using trained model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/trained_model.joblib",
        help="Path to trained model file (default: models/trained_model.joblib)",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        help="Feature values: sepal_length sepal_width petal_length petal_width",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of test samples to use if features not provided (default: 3)",
    )

    args = parser.parse_args()

    features_list = None
    if args.features:
        try:
            features_list = [float(x) for x in args.features]
        except ValueError:
            print("Error: All features must be numeric")
            sys.exit(1)

    predict(model_path=args.model, features=features_list, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
