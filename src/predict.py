"""
Prediction script for end-to-end ML project.
"""

import argparse
from pathlib import Path

import numpy as np
from joblib import load
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():
    """Load a trained model and make predictions on new data."""
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

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = load(model_path)
    print("Model loaded successfully!")

    # Load target names for display
    iris = load_iris()
    target_names = iris.target_names

    # Get input features
    if args.features:
        try:
            features = [float(x) for x in args.features]
            if len(features) != 4:
                print("Error: Please provide exactly 4 feature values")
                return
            X = np.array([features])
        except ValueError:
            print("Error: All features must be numeric")
            return
    else:
        # Use test samples from dataset as example
        print("\nNo features provided, using sample from test set...")
        _, X_test, _, _ = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        X = X_test[:3]  # Use first 3 test samples

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
    main()

