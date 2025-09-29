import torch
import os
import joblib

from src.dataset import load_dataset
from src.train import train_logistic_regression, train_random_forest
from src.utils import evaluate_logistic, evaluate_random_forest, display_confusion_matrix
from src.models import LogisticRegressionModel

def print_metrics(name, metrics):
    """Helper to format and print evaluation metrics."""
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

def run_with_training():
    """Train both the models, save them, and evaluate them."""
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_dataset("data/customer_churn_features.csv")

    # Train Logistic Regression (PyTorch)
    input_dim = train_dataset.x.shape[1]
    print("\nTraining Logistic Regression (PyTorch)...")
    logistic_model = train_logistic_regression(train_dataset, val_dataset, input_dim, device=device)

    os.makedirs("models", exist_ok=True)
    torch.save(logistic_model.state_dict(), "models/logistic_regression.pt")

    # Train Random Forest
    print("\nTraining Random Forest (sklearn)...")
    rf_model = train_random_forest(train_dataset, val_dataset)
    joblib.dump(rf_model, "models/random_forest.pkl")

    # Evaluation
    print("\nEvaluating models on test set...")
    log_metrics = evaluate_logistic(logistic_model, test_dataset, device=device)
    rf_metrics = evaluate_random_forest(rf_model, test_dataset)

    print_metrics("Logistic Regression (PyTorch)", log_metrics)
    print_metrics("Random Forest (sklearn)", rf_metrics)

    # Confusion Matrix for Best Model
    display_confusion_matrix(log_metrics, rf_metrics)


def run_existing():
    """Load already trained models and evaluate them."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load datasets
    _, _, test_dataset = load_dataset("data/customer_churn_features.csv")
    logistic_path = "models/logistic_regression.pt"
    random_forest_path = "models/random_forest.pkl"

    if not os.path.exists(logistic_path) or not os.path.exists(random_forest_path):
        print("\n⚠Saved models not found. Training new models instead.")
        run_with_training()
        return

    # Load Logistic Regression (PyTorch)
    input_dim = test_dataset.x.shape[1]
    logistic_model = LogisticRegressionModel(input_dim).to(device)
    logistic_model.load_state_dict(torch.load(logistic_path, map_location=device))

    # Load Random Forest
    rf_model = joblib.load(random_forest_path)

    # Evaluate
    log_metrics = evaluate_logistic(logistic_model, test_dataset, device=device)
    rf_metrics = evaluate_random_forest(rf_model, test_dataset)

    print_metrics("Logistic Regression (PyTorch)", log_metrics)
    print_metrics("Random Forest (sklearn)", rf_metrics)

    # Confusion Matrix
    display_confusion_matrix(log_metrics, rf_metrics)


if __name__ == "__main__":
    run_mode = "train" # change to use existing model
    if run_mode == "train":
        run_with_training()
    else:
        run_existing()

