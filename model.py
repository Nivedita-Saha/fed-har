"""
model.py — the small neural network shared by all experiments.

Feature vector (561 numbers) goes in; six activity scores come out.
Deliberately simple: the novelty in this project is the federated
training, not the architecture.
"""

import torch.nn as nn


class HARNet(nn.Module):
    def __init__(self, n_features=561, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)