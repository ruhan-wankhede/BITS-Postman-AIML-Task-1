import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_logistic(model, test_dataset, device=None):
    """Evaluate the PyTorch logistic regression model on the test set."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    x_test = test_dataset.x.to(device)
    y_test = test_dataset.y.to(device)

    with torch.no_grad():
        outputs = model(x_test).cpu()
        preds = (outputs > 0.5).int().squeeze()

    y_true = y_test.cpu().int()
    y_pred = preds

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }

    return metrics


def evaluate_random_forest(model, test_dataset):
    """Evaluate the Random Forest model on the test set."""
    x_test = test_dataset.x.cpu().numpy()
    y_true = test_dataset.y.cpu().numpy()
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }

    return metrics


def display_confusion_matrix(logistic_metrics, rf_metrics):
    """Display confusion matrix for the better model based on F1-score."""
    if logistic_metrics["f1"] >= rf_metrics["f1"]:
        best = "Logistic Regression (PyTorch)"
        y_true = logistic_metrics["y_true"]
        y_pred = logistic_metrics["y_pred"]
    else:
        best = "Random Forest (sklearn)"
        y_true = rf_metrics["y_true"]
        y_pred = rf_metrics["y_pred"]

    print(f"\nBest model based on F1-score: {best}")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Active", "Churn"], yticklabels=["Active", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({best})")
    plt.show()
