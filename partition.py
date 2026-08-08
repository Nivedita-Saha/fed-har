"""
partition.py — split the training data into federated clients two ways.

by_person : one client per training person (21 clients, uneven sizes).
            Realistic and non-independent, because each person moves
            differently — the hard, interesting case.

even      : shuffle all rows together and deal them into equal random
            piles. Independent and balanced — the easy case.

Each returns a dict: client_id -> (X_client, y_client).
"""

import numpy as np

from data_loader import load_split


def partition_by_person():
    X, y, subjects = load_split("train")
    clients = {}
    for person in sorted(np.unique(subjects)):
        mask = subjects == person
        clients[int(person)] = (X[mask], y[mask])
    return clients


def partition_even(n_clients=21, seed=42):
    X, y, _ = load_split("train")
    rng = np.random.default_rng(seed)

    order = rng.permutation(len(X))       # shuffle row indices
    shards = np.array_split(order, n_clients)   # deal into equal piles

    clients = {}
    for i, idx in enumerate(shards):
        clients[i] = (X[idx], y[idx])
    return clients


if __name__ == "__main__":
    by_person = partition_by_person()
    even = partition_even()

    print("BY-PERSON split:", len(by_person), "clients")
    sizes = {cid: len(X) for cid, (X, y) in by_person.items()}
    print("   rows per client:", sizes)
    print("   min", min(sizes.values()), " max", max(sizes.values()))
    print()

    print("EVEN split:", len(even), "clients")
    sizes = {cid: len(X) for cid, (X, y) in even.items()}
    print("   rows per client:", sizes)
    print("   min", min(sizes.values()), " max", max(sizes.values()))