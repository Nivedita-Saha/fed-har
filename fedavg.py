"""
fedavg.py — FedAvg (McMahan et al., 2017) implemented from scratch.

Each round:
  1. server picks a random fraction of clients
  2. each picked client trains a private copy on ITS OWN data only
  3. server averages the returned weights (weighted by data size)
  4. that average becomes the new shared model; test it on the global set

Run:  python fedavg.py even
      python fedavg.py by_person
"""

import copy
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_loader import load_split
from model import HARNet
from partition import partition_by_person, partition_even

# ---- federated settings ----
ROUNDS = 50            # communication rounds
CLIENT_FRACTION = 0.3  # fraction of clients sampled each round
LOCAL_EPOCHS = 3       # local passes each client makes per round
LOCAL_BATCH = 32
LOCAL_LR = 1e-3
SEED = 42


def make_loader(X, y, batch_size, shuffle):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y - 1, dtype=torch.long)   # labels 1-6 -> 0-5
    return DataLoader(TensorDataset(X_t, y_t),
                      batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


def local_train(global_model, X, y):
    """Client makes a private copy, trains on its own data, returns weights."""
    model = copy.deepcopy(global_model)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=LOCAL_LR)
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(X, y, LOCAL_BATCH, shuffle=True)

    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()

    return model.state_dict()


def average_weights(weight_list, sizes):
    """Weighted average of client weights (bigger clients count more)."""
    total = float(sum(sizes))
    avg = copy.deepcopy(weight_list[0])
    for key in avg.keys():
        avg[key] = sum(w[key] * (n / total)
                       for w, n in zip(weight_list, sizes))
    return avg


def run(split_name):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    if split_name == "by_person":
        clients = partition_by_person()
    elif split_name == "even":
        clients = partition_even()
    else:
        raise SystemExit("split must be 'even' or 'by_person'")

    client_ids = list(clients.keys())
    n_per_round = max(1, int(CLIENT_FRACTION * len(client_ids)))

    X_test, y_test, _ = load_split("test")
    test_loader = make_loader(X_test, y_test, 64, shuffle=False)

    global_model = HARNet()
    history = []

    for rnd in range(1, ROUNDS + 1):
        chosen = rng.choice(client_ids, size=n_per_round, replace=False)

        weights, sizes = [], []
        for cid in chosen:
            X, y = clients[cid]
            weights.append(local_train(global_model, X, y))
            sizes.append(len(X))

        global_model.load_state_dict(average_weights(weights, sizes))

        acc = evaluate(global_model, test_loader)
        history.append(acc)
        if rnd % 5 == 0 or rnd == 1:
            print(f"round {rnd:2d}  test accuracy {acc:.4f}")

    print()
    print(f"FEDAVG ({split_name}) FINAL: {history[-1]:.4f}")

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"fedavg_{split_name}.json"
    out.write_text(json.dumps({"method": "fedavg",
                               "split": split_name,
                               "history": history}))
    print("saved", out)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "even"
    run(split)