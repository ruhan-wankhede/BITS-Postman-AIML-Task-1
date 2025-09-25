import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ChurnDataset(Dataset):
    def __init__(self, features, labels):
        # store data as tensors
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels.values, dtype=torch.float32) # churn is only 0 or 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_dataset(dataset_path: str, test_size: float = 0.15, validation_size: float = 0.15, random_state: int = 67):
    """
    Loading the saved dataset after eda and rfm feature engineering in the jupyter notebook.
    :param dataset_path: Path to the dataset
    :param test_size: Size of the test set
    :param validation_size: Size of the validation set
    :param random_state: Random seed for the train test validation splits
    :return: None
    """

    df = pd.read_csv(dataset_path)

    x = df[["recency", "frequency", "monetary_value"]]
    y = df["churn"]

    # First split: train vs temp (val+test)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=(test_size + validation_size), stratify=y, random_state=random_state
    )

    # Second split: validation vs test
    relative_test_size = test_size / (test_size + validation_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=relative_test_size, stratify=y_temp, random_state=random_state
    )

    # Scaling the features since monetary value and frequency have differing magnitudes
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    train_dataset = ChurnDataset(x_train_scaled, y_train)
    val_dataset = ChurnDataset(x_val_scaled, y_val)
    test_dataset = ChurnDataset(x_test_scaled, y_test)

    return train_dataset, val_dataset, test_dataset