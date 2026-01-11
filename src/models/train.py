"""Model training module."""

from pathlib import Path

from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.data import load_iris_data
from src.features import standardize_features


def train_model(
    test_size: float = 0.2,
    random_state: int = 42,
    standardize: bool = False,
    model_path: str = "models/trained_model.joblib",
) -> LogisticRegression:
    """
    Train and evaluate a logistic regression model on the Iris dataset.

    Args:
        test_size: Proportion of dataset to include in test split
        random_state: Random state for reproducibility
        standardize: Whether to standardize features
        model_path: Path to save the trained model

    Returns:
        Trained LogisticRegression model
    """
    print("Loading Iris dataset...")
    X, y, target_names = load_iris_data()
    print(f"Dataset shape: {X.shape}")

    # Split data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Feature engineering (optional standardization)
    if standardize:
        print("Standardizing features...")
        X_train, X_test, scaler = standardize_features(X_train, X_test)
    else:
        scaler = None

    # Train model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=200, random_state=random_state)
    model.fit(X_train, y_train)
    print("Model training completed!")

    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save model
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Save scaler if used
    if scaler is not None:
        scaler_path = Path(model_path).parent / "scaler.joblib"
        dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    return model


if __name__ == "__main__":
    train_model()
