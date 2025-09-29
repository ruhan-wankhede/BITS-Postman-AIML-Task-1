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

    df = pd.read_csv(dataset_path, parse_dates=["order_purchase_timestamp"])

    #  Split customers first to avoid data leakage
    unique_customers = df["customer_unique_id"].unique()
    train_ids, temp_ids = train_test_split(unique_customers, test_size=(test_size + validation_size),
                                           random_state=random_state)
    val_ids, test_ids = train_test_split(temp_ids, test_size=test_size / (test_size + validation_size),
                                         random_state=random_state)

    train_orders = df[df["customer_unique_id"].isin(train_ids)]
    val_orders = df[df["customer_unique_id"].isin(val_ids)]
    test_orders = df[df["customer_unique_id"].isin(test_ids)]

    reference_date = train_orders["order_purchase_timestamp"].max()

    def build_features(order_df, ref_date):
        features = (
            order_df.groupby("customer_unique_id")
            .agg(
                last_purchase=("order_purchase_timestamp", "max"),
                frequency=("order_id", "count"),
                monetary_value=("payment_value", "sum")
            )
            .reset_index()
        )
        features["recency"] = (ref_date - features["last_purchase"]).dt.days
        features["churn"] = (features["recency"] > 180).astype(int)
        return features[["customer_unique_id", "recency", "frequency", "monetary_value", "churn"]]

    train_features = build_features(train_orders, reference_date)
    val_features = build_features(val_orders, reference_date)
    test_features = build_features(test_orders, reference_date)

    x_train, y_train = train_features[["recency", "frequency", "monetary_value"]], train_features["churn"]
    x_val, y_val = val_features[["recency", "frequency", "monetary_value"]], val_features["churn"]
    x_test, y_test = test_features[["recency", "frequency", "monetary_value"]], test_features["churn"]


    # Scaling the features since monetary value and frequency have differing magnitudes
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    train_dataset = ChurnDataset(x_train_scaled, y_train)
    val_dataset = ChurnDataset(x_val_scaled, y_val)
    test_dataset = ChurnDataset(x_test_scaled, y_test)

    return train_dataset, val_dataset, test_dataset