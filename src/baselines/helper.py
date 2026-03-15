"""
Simple baseline models for paraphrase detection.
Implements Random and Majority baselines per project instructions.
"""

import random
import numpy as np
from sklearn.metrics import mean_squared_error

# Data path
TRAIN_RATIO = 0.70
TEST_RATIO = 0.20
VAL_RATIO = 0.10
RANDOM_SEED = 42


def load_data(filepath):
    """Load para-nmt format: sentence1, sentence2, score (tab-separated)."""
    sentences1 = []
    sentences2 = []
    scores = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                sentences1.append(parts[0])
                sentences2.append(parts[1])
                scores.append(float(parts[2]))
    return sentences1, sentences2, np.array(scores)


def split_data(scores):
    """Split into 70% train, 20% test, 10% validation."""
    n = len(scores)
    indices = list(range(n))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    n_train = int(n * TRAIN_RATIO)
    n_test = int(n * TEST_RATIO)

    train_idx = indices[:n_train]
    test_idx = indices[n_train : n_train + n_test]
    val_idx = indices[n_train + n_test :]

    y_train = scores[train_idx]
    y_test = scores[test_idx]
    y_val = scores[val_idx]

    return y_train, y_test, y_val, test_idx, val_idx


def evaluate(y_true, y_pred, name):
    """Compute MSE and accuracy (1 - MSE) as in project scoring."""
    mse = mean_squared_error(y_true, y_pred)
    accuracy = (1.0 - mse) * 100.0
    print(f"\n{name}:")
    print(f"  MSE:      {mse:.6f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    return mse, accuracy