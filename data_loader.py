"""
data_loader.py — load the UCI HAR dataset for federated learning.

Each row is one short window of phone-sensor features (561 of them),
tagged with an activity label (1-6) and the person it came from (1-30).

We keep the official train/test split:
  - training people  -> become our federated clients
  - test people      -> become the global test set (no client trains on it)
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/UCI HAR Dataset")


def _read_matrix(path):
    """Read a whitespace-separated numeric file into a numpy array."""
    return pd.read_csv(path, sep=r"\s+", header=None).values


def _read_column(path):
    """Read a single-column file into a flat numpy array."""
    return pd.read_csv(path, sep=r"\s+", header=None).values.ravel()


def load_split(split):
    """Load one split ('train' or 'test'): features, labels, subject ids."""
    X = _read_matrix(DATA_DIR / split / f"X_{split}.txt")
    y = _read_column(DATA_DIR / split / f"y_{split}.txt")
    subjects = _read_column(DATA_DIR / split / f"subject_{split}.txt")
    return X, y, subjects


def load_activity_names():
    """Map activity id (1-6) to its human-readable name."""
    df = pd.read_csv(DATA_DIR / "activity_labels.txt", sep=r"\s+",
                     header=None, index_col=0)
    return df[1].to_dict()


if __name__ == "__main__":
    X_train, y_train, subj_train = load_split("train")
    X_test, y_test, subj_test = load_split("test")
    activities = load_activity_names()

    print("TRAIN:", X_train.shape[0], "rows,", X_train.shape[1], "features")
    print("TEST: ", X_test.shape[0], "rows,", X_test.shape[1], "features")
    print()

    print("Activities:")
    for k, v in activities.items():
        print("  ", k, "=", v)
    print()

    print("Unique labels in train:", sorted(np.unique(y_train)))
    print("Unique labels in test: ", sorted(np.unique(y_test)))
    print()

    train_people = sorted(int(p) for p in np.unique(subj_train))
    test_people = sorted(int(p) for p in np.unique(subj_test))
    print("Training people (clients):", len(train_people), "->", train_people)
    print("Test people:", len(test_people), "->", test_people)
    print()

    print("Rows per training person:")
    for p in train_people:
        print("   person", p, ":", int((subj_train == p).sum()), "rows")