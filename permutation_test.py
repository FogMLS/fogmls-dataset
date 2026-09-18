# -*- coding: utf-8 -*-
"""
permutation_test.py

Leakage diagnostic: loads a trained RF model, gets its predictions on the
real held-out test set, then shuffles the TRUE test labels (breaking any
real correspondence between features and labels) and checks how well the
model's (unchanged) predictions match the shuffled labels.

If accuracy under permutation drops close to chance level (roughly the
squared sum of class frequencies), the model's real accuracy reflects
genuine learned signal. If accuracy stays close to the real accuracy,
that is a warning sign of leakage (something other than the label is
letting the model "guess right").

Usage:
    python3 permutation_test.py <dataset_file> <model.joblib> [n_shuffles] [results_log.csv]
"""

import sys
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

MAX_JOBS = 200
JOB_ATTRIBUTES = 6
SLOT_LEVEL_FEATURES = 11


def parse_and_reshape(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    X = data[:, :MAX_JOBS * JOB_ATTRIBUTES].reshape(-1, MAX_JOBS, JOB_ATTRIBUTES)
    y_start = MAX_JOBS * JOB_ATTRIBUTES + SLOT_LEVEL_FEATURES
    Y = data[:, y_start:y_start + MAX_JOBS]

    rows_X, rows_Y = [], []
    for s in range(X.shape[0]):
        for j in range(MAX_JOBS):
            task = X[s, j]
            label = Y[s, j]
            if task.sum() == 0 and label == 0:
                continue
            rows_X.append(np.append(task, j))
            rows_Y.append(label)

    return np.array(rows_X), np.array(rows_Y, dtype=int)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 permutation_test.py <dataset_file> <model.joblib> [n_shuffles] [results_log.csv]")
        sys.exit(1)

    filepath = sys.argv[1]
    model_path = sys.argv[2]
    n_shuffles = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    results_log = sys.argv[4] if len(sys.argv) > 4 else None

    print(f"[PERM] Loading and reshaping {filepath}...")
    X, Y = parse_and_reshape(filepath)

    split = int(0.8 * len(X))
    X_test, Y_test = X[split:], Y[split:]
    print(f"[PERM] Test set: {len(X_test)} rows")

    print(f"[PERM] Loading model {model_path}...")
    rf = joblib.load(model_path)

    pred = rf.predict(X_test)
    real_acc = accuracy_score(Y_test, pred)
    print(f"[PERM] Real (unpermuted) test accuracy: {100*real_acc:.2f}%")

    names = ['Nothing', 'Cloud', 'Fog-1', 'Fog-2', 'Fog-3']
    print()
    print("Real classification report:")
    print(classification_report(Y_test, pred, target_names=names,
                                 labels=[0, 1, 2, 3, 4], zero_division=0))

    rng = np.random.RandomState(42)
    perm_accs = []
    print(f"[PERM] Running {n_shuffles} independent label permutations...")
    for i in range(n_shuffles):
        Y_shuffled = rng.permutation(Y_test)
        perm_acc = accuracy_score(Y_shuffled, pred)
        perm_accs.append(perm_acc)

    perm_accs = np.array(perm_accs)
    print()
    print("=" * 50)
    print("  PERMUTATION TEST RESULTS")
    print("=" * 50)
    print(f"  Real accuracy:        {100*real_acc:.2f}%")
    print(f"  Permuted accuracy:    {100*perm_accs.mean():.2f}% "
          f"(std {100*perm_accs.std():.2f}%, over {n_shuffles} shuffles)")
    print(f"  Drop:                 {100*(real_acc - perm_accs.mean()):.2f} percentage points")

    Y_shuffled_example = rng.permutation(Y_test)
    print()
    print("Example single-shuffle classification report (for per-class F1 table):")
    print(classification_report(Y_shuffled_example, pred, target_names=names,
                                 labels=[0, 1, 2, 3, 4], zero_division=0))

    if results_log:
        with open(results_log, 'a') as f:
            f.write(f"{filepath},{model_path},{100*real_acc:.4f},"
                    f"{100*perm_accs.mean():.4f},{100*perm_accs.std():.4f}\n")
        print(f"[PERM] Logged to {results_log}")


if __name__ == '__main__':
    main()
