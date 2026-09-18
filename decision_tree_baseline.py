# -*- coding: utf-8 -*-
"""
decision_tree_baseline.py

Decision Tree baseline using the SAME task-level feature representation
as random_forest_baseline.py (six task features + queue position), for
a fair, apples-to-apples comparison against the Random Forest model.

Usage:
    python3 decision_tree_baseline.py <dataset_file> <max_depth> [results_log.csv]
"""

import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
        print("Usage: python3 decision_tree_baseline.py <dataset_file> <max_depth> [results_log.csv]")
        sys.exit(1)

    filepath = sys.argv[1]
    max_depth = int(sys.argv[2])
    results_log = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"[DT] Loading and reshaping {filepath}...")
    X, Y = parse_and_reshape(filepath)
    print(f"[DT] Task-level dataset: {X.shape[0]} rows, {X.shape[1]} features")

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    print(f"[DT] Training Decision Tree (max_depth={max_depth})...")
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, Y_train)

    pred = dt.predict(X_test)
    acc = accuracy_score(Y_test, pred)

    print(f"[DT] Accuracy (depth={max_depth}): {100*acc:.2f}%")

    if results_log:
        with open(results_log, 'a') as f:
            f.write(f"{filepath},{max_depth},{100*acc:.4f}\n")
        print(f"[DT] Logged to {results_log}")


if __name__ == '__main__':
    main()
