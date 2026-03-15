"""
Simple baseline models for paraphrase detection.
Implements Random and Majority baselines per project instructions.
"""

import random
import numpy as np
import helper


DATA_PATH = "src/data/para-nmt-50m/para-nmt-1m.txt"


class RandomBaseline:
    """Randomly assigns a score to each data point."""

    def fit(self, y_train):
        self.y_train = y_train
        return self

    def predict(self, n):
        """Predict n random scores (uniform 0-1)."""
        random.seed(helper.RANDOM_SEED)
        return np.array([random.uniform(0, 1) for _ in range(n)])


class MajorityBaseline:
    """Always predicts the most common value (mean for continuous scores)."""

    def fit(self, y_train):
        self.prediction = np.mean(y_train)
        return self

    def predict(self, n):
        """Predict the mean for all n samples."""
        return np.full(n, self.prediction)


def main():
    # Load data
    sentences1, sentences2, scores = helper.load_data(DATA_PATH)

    # Split data
    y_train, y_test, y_val, test_idx, val_idx = helper.split_data(scores)

    # Random baseline
    random_baseline = RandomBaseline()
    random_baseline.fit(y_train)
    random_pred_test = random_baseline.predict(len(y_test))
    random_pred_val = random_baseline.predict(len(y_val))

    # Majority baseline
    majority_baseline = MajorityBaseline()
    majority_baseline.fit(y_train)
    majority_pred_test = majority_baseline.predict(len(y_test))
    majority_pred_val = majority_baseline.predict(len(y_val))

    # Evaluate on test set
    print("Test set results:")
    random_mse, random_acc = helper.evaluate(y_test, random_pred_test, "Random Baseline")
    majority_mse, majority_acc = helper.evaluate(y_test, majority_pred_test, "Majority Baseline")
    print("\n" * 2)

    # Evaluate on validation set
    print("Validation set results:")
    helper.evaluate(y_val, random_pred_val, "Random Baseline")
    helper.evaluate(y_val, majority_pred_val, "Majority Baseline")


if __name__ == "__main__":
    main()
