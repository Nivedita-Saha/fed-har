"""
centralised.py — the centralised baseline (upper bound).

Pool ALL training people's data together, train the network normally,
and measure accuracy on the held-out test people. This is the target
every federated configuration is compared against.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_loader import load_split
from model import HARNet

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
SEED = 42


def make_loader(X, y, batch_size, shuffle):
    # labels are 1-6 in the file; shift to 0-5 for PyTorch
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y - 1, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train, _ = load_split("train")
    X_test, y_test, _ = load_split("test")

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    test_loader = make_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

    model = HARNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader)
            print(f"epoch {epoch:2d}  test accuracy {acc:.4f}")

    final_acc = evaluate(model, test_loader)
    print()
    print(f"CENTRALISED UPPER BOUND: {final_acc:.4f}")


if __name__ == "__main__":
    main()