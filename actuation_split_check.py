# -*- coding: utf-8 -*-
"""
actuation_split_check.py

Checks whether accuracy differs between actuation tasks (zero instruction
count, zero data size) and real tasks (sensing/processing, non-zero
features), using an already-trained model.

Usage:
    python3 actuation_split_check.py <dataset_file> <model.joblib>
"""

import sys
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

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

    rows_X, rows_Y, rows_tasktype = [], [], []
    for s in range(X.shape[0]):
        for j in range(MAX_JOBS):
            task = X[s, j]
            label = Y[s, j]
            if task.sum() == 0 and label == 0:
                continue
            rows_X.append(np.append(task, j))
            rows_Y.append(label)
            rows_tasktype.append(task[0])

    rows_X = np.array(rows_X)
    rows_Y = np.array(rows_Y, dtype=int)
    rows_tasktype = np.array(rows_tasktype)

    split = int(0.8 * len(rows_X))
    X_test = rows_X[split:]
    Y_test = rows_Y[split:]
    tt_test = rows_tasktype[split:]

    rf = joblib.load(model_path)
    pred = rf.predict(X_test)

    overall_acc = accuracy_score(Y_test, pred)
    print(f'Overall test accuracy: {100*overall_acc:.2f}%')

    is_actuation = (tt_test == 4) | (tt_test == 5)
    is_real = ~is_actuation

    acc_actuation = accuracy_score(Y_test[is_actuation], pred[is_actuation])
    acc_real = accuracy_score(Y_test[is_real], pred[is_real])

    print(f'Actuation-task accuracy (n={is_actuation.sum()}): {100*acc_actuation:.2f}%')
    print(f'Real-task accuracy (n={is_real.sum()}): {100*acc_real:.2f}%')
    print(f'Actuation fraction of test set: {100*is_actuation.sum()/len(tt_test):.1f}%')


if __name__ == '__main__':
    main()
