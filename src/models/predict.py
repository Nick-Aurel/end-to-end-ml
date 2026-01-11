"""Model prediction module."""

from pathlib import Path
from typing import List, Optional

import numpy as np
from joblib import load
from sklearn.datasets import load_iris

from src.data import load_iris_data


def predict(
    model_path: str = "models/trained_model.joblib",
    features: Optional[List[float]] = None,
    num_samples: int = 3,
) -> None:
    """
    Load a trained model and make predictions on new data.

    Args:
        model_path: Path to trained model file
        features: Optional list of feature values (4 values for Iris dataset)
        num_samples: Number of test samples to use if features not provided
    """
    # Load model
    model_filepath = Path(model_path)
    if not model_filepath.exists():
        print(f"Error: Model file not found at {model_filepath}")
        return

    print(f"Loading model from {model_filepath}...")
    model = load(model_filepath)
    print("Model loaded successfully!")

    # Load scaler if exists
    scaler_path = model_filepath.parent / "scaler.joblib"
    scaler = None
    if scaler_path.exists():
        scaler = load(scaler_path)
        print("Scaler loaded successfully!")

    # Load target names for display
    _, _, target_names = load_iris_data()

    # Get input features
    if features:
        if len(features) != 4:
            print("Error: Please provide exactly 4 feature values")
            return
        X = np.array([features])
    else:
        # Use test samples from dataset as example
        print(f"\nNo features provided, using {num_samples} sample(s) from test set...")
        iris = load_iris()
        from sklearn.model_selection import train_test_split

        _, X_test, _, _ = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        X = X_test[:num_samples]

    # Apply scaler if available
    if scaler is not None:
        X = scaler.transform(X)

    # Make predictions
    print(f"\nMaking predictions on {len(X)} sample(s)...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Display results
    print("\nPredictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        class_name = target_names[pred]
        confidence = prob[pred] * 100
        print(f"\nSample {i + 1}:")
        print(f"  Features: {X[i]}")
        print(f"  Predicted class: {class_name}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Probabilities: {dict(zip(target_names, prob))}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make predictions using trained model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/trained_model.joblib",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        help="Feature values: sepal_length sepal_width petal_length petal_width",
    )

    args = parser.parse_args()

    features_list = None
    if args.features:
        try:
            features_list = [float(x) for x in args.features]
        except ValueError:
            print("Error: All features must be numeric")
            exit(1)

    predict(model_path=args.model, features=features_list)
