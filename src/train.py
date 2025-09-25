import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from models import LogisticRegressionModel


def train_logistic_regression(train_dataset, val_dataset,input_dim, epochs=50, batch_size=32, lr=0.001, device=None) -> LogisticRegressionModel:
    """
    Train the logistic regression model
    :param train_dataset: the training dataset
    :param val_dataset: the validation dataset
    :param input_dim: the dimensions of the input
    :param epochs:
    :param batch_size:
    :param lr: learning rate
    :param device: "cuda" | "cpu" | None (auto detect)
    :return: returns the trained logistic regression model
    """

    # Auto-detect device if not given
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)  # full-batch val

    # Initialize model
    model = LogisticRegressionModel(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for val_x, val_y in val_loader:  # only 1 batch here
                    val_x, val_y = val_x.to(device), val_y.to(device).view(-1, 1)
                    val_preds = model(val_x).cpu()
                    val_preds = (val_preds > 0.5).int()
                    acc = accuracy_score(val_y.cpu(), val_preds)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Acc: {acc:.4f}")

    return model


def train_random_forest(train_dataset, val_dataset) -> RandomForestClassifier:
    """
    Train the random forest model using scikit learn
    :param train_dataset:
    :param val_dataset:
    :return:
    """
    # Convert datasets back to numpy
    x_train = train_dataset.x.cpu().numpy()
    y_train = train_dataset.y.cpu().numpy()
    x_val = val_dataset.x.cpu().numpy()
    y_val = val_dataset.y.cpu().numpy()

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=67,
        n_jobs=-1
    )
    clf.fit(x_train, y_train)

    # Validation performance
    y_val_pred = clf.predict(x_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))

    return clf