import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    """Simple Logistic regression model"""
    def __init__(self, input_dim: int):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1) # output is only 0 or 1 logit

    def forward(self, x) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))