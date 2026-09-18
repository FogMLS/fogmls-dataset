# -*- coding: utf-8 -*-
"""
cross_seed_eval.py

Loads a Random Forest model trained on one seed's dataset and evaluates
it on a DIFFERENT seed's full dataset (no retraining), to test whether
learned scheduling patterns generalize to an unseen workload trajectory.

Usage:
    python3 cross_seed_eval.py <model.joblib> <other_seed_dataset_file> [results_log.csv]
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
        print("Usage: python3 cross_seed_eval.py <model.joblib> <other_seed_dataset_file> [results_log.csv]")
        sys.exit(1)

    model_path = sys.argv[1]
    other_seed_file = sys.argv[2]
    results_log = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"[CROSS-SEED] Loading model {model_path}...")
    rf = joblib.load(model_path)

    print(f"[CROSS-SEED] Loading and reshaping {other_seed_file}...")
    X, Y = parse_and_reshape(other_seed_file)
    print(f"[CROSS-SEED] Full dataset (no train/test split, evaluating on ALL of it): "
          f"{X.shape[0]} rows, {X.shape[1]} features")

    pred = rf.predict(X)
    acc = accuracy_score(Y, pred)

    print()
    print("=" * 50)
    print(f"  CROSS-SEED RESULTS: {model_path} -> {other_seed_file}")
    print("=" * 50)
    print(f"  Cross-seed accuracy: {100*acc:.2f}%")
    print()
    names = ['Nothing', 'Cloud', 'Fog-1', 'Fog-2', 'Fog-3']
    print(classification_report(Y, pred, target_names=names,
                                 labels=[0, 1, 2, 3, 4], zero_division=0))

    if results_log:
        with open(results_log, 'a') as f:
            f.write(f"{model_path},{other_seed_file},{100*acc:.4f}\n")
        print(f"[CROSS-SEED] Logged to {results_log}")


if __name__ == '__main__':
    main()
