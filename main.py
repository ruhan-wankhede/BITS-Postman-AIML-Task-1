from torch.utils.data import DataLoader
from src.dataset import load_dataset

# Load train/test datasets
train_dataset, validation_dataset, test_dataset = load_dataset("data/customer_churn_features.csv")

# DataLoaders (batch training)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)