# -*- coding: utf-8 -*-
"""
run_rf_named.py

Thin wrapper around random_forest_baseline.py's parse_and_reshape() and
RandomForestClassifier training, but saves the model under a caller-chosen
filename instead of always overwriting random_forest_model.joblib. This
lets multiple scenario/seed runs (low/medium/high load, seeds 42/123/456)
keep their own trained models around for later cross-seed evaluation,
without needing to touch the original random_forest_baseline.py.

Usage:
    python3 run_rf_named.py <dataset_file> <model_output_name.joblib> [results_log.csv]

Appends one line to results_log.csv (if given) in the form:
    dataset_file,model_output_name,accuracy
"""

import sys
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
        print("Usage: python3 run_rf_named.py <dataset_file> <model_output_name.joblib> [results_log.csv]")
        sys.exit(1)

    filepath = sys.argv[1]
    model_out = sys.argv[2]
    results_log = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"[RF] Loading and reshaping {filepath}...")
    X, Y = parse_and_reshape(filepath)
    print(f"[RF] Task-level dataset: {X.shape[0]} rows, {X.shape[1]} features")

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    print(f"[RF] Train: {len(X_train)}, Test: {len(X_test)}")

    print("[RF] Training Random Forest (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=25,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, Y_train)
    joblib.dump(rf, model_out)
    print(f"[RF] Model saved to {model_out}")

    pred = rf.predict(X_test)
    acc = accuracy_score(Y_test, pred)

    print()
    print("=" * 50)
    print(f"  RESULTS for {filepath}")
    print("=" * 50)
    print(f"  Overall accuracy: {100*acc:.2f}%")
    print()
    names = ['Nothing', 'Cloud', 'Fog-1', 'Fog-2', 'Fog-3']
    print(classification_report(Y_test, pred, target_names=names,
                                 labels=[0, 1, 2, 3, 4], zero_division=0))

    if results_log:
        with open(results_log, 'a') as f:
            f.write(f"{filepath},{model_out},{100*acc:.4f}\n")
        print(f"[RF] Logged to {results_log}")


if __name__ == '__main__':
    main()
