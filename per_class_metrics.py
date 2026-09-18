# -*- coding: utf-8 -*-
"""
per_class_metrics.py

Prints per-class precision/recall/F1 for a trained model on its own
dataset's held-out test split (same 80/20 split as training).

Usage:
    python3 per_class_metrics.py <dataset_file> <model.joblib>
"""

import sys
import joblib
import numpy as np
from sklearn.metrics import classification_report

MAX_JOBS = 200
JOB_ATTRIBUTES = 6
SLOT_LEVEL_FEATURES = 11


def main():
    filepath = sys.argv[1]
    model_path = sys.argv[2]

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

    rows_X = np.array(rows_X)
    rows_Y = np.array(rows_Y, dtype=int)

    split = int(0.8 * len(rows_X))
    X_test = rows_X[split:]
    Y_test = rows_Y[split:]

    rf = joblib.load(model_path)
    pred = rf.predict(X_test)

    names = ['Cloud', 'Fog-1', 'Fog-2', 'Fog-3']
    print(classification_report(Y_test, pred, labels=[1, 2, 3, 4],
                                 target_names=names, digits=4))


if __name__ == '__main__':
    main()
