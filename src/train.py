"""
Training script for end-to-end ML project.
"""

from pathlib import Path

from joblib import dump
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main():
    """Train and evaluate a logistic regression model on the Iris dataset."""
    print("Loading Iris dataset...")
    data = load_iris()
    X, y = data.data, data.target
    print(f"Dataset shape: {X.shape}")

    # Split data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed!")

    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "trained_model.joblib"
    dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
