# -*- coding: utf-8 -*-
"""
data_loader.py - FogMLS Dataset Loader
Loads the FogMLS dataset from Zenodo, applies StandardScaler
normalization, and returns an 80/20 stratified train/test split.

Usage:
    from data_loader import load_dataset
    X_train, X_test, Y_train, Y_test = load_dataset('fogmls_dataset_seed42.txt')

Dataset download:
    https://doi.org/10.5281/zenodo.19382240
"""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ================================================================
# DATASET CONSTANTS
# Fixed values for the FogMLS seed-42 dataset.
# Do not change these unless you regenerate the dataset.
# ================================================================
MAX_JOBS        = 200   # Task positions per slot
MAX_RS          = 5     # Resource classes: Nothing, Cloud, Fog-1, Fog-2, Fog-3
FEATURES_PER_JOB = 3   # TaskType, TaskCodeSize, TaskDataSize

RANDOM_STATE    = 42
TEST_SIZE       = 0.2
SCALER_FILE     = 'DL_scaler_200.joblib'


def parse_dataset(fname):
    """
    Reads the FogMLS dataset CSV and returns X (features) and Y (labels).

    Each row represents one simulation slot.
    X columns: MAX_JOBS x FEATURES_PER_JOB flat feature vector.
    Y columns: MAX_JOBS resource assignment labels (0=Nothing,
               1=Cloud, 2=Fog-1, 3=Fog-2, 4=Fog-3).
    """
    if not os.path.exists(fname):
        print("[DATA] File not found: " + fname)
        sys.exit(1)

    print("[DATA] Loading: " + fname)
    rows_X, rows_Y = [], []

    with open(fname, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(c.isalpha() for c in line):
            continue
        try:
            values = [float(v.strip()) for v in line.split(',')
                      if v.strip() != '']
        except ValueError:
            continue
        expected = MAX_JOBS * FEATURES_PER_JOB + MAX_JOBS
        if len(values) < expected:
            continue
        rows_X.append(values[:MAX_JOBS * FEATURES_PER_JOB])
        rows_Y.append(values[MAX_JOBS * FEATURES_PER_JOB:
                              MAX_JOBS * FEATURES_PER_JOB + MAX_JOBS])

    if len(rows_X) == 0:
        print("[DATA] No valid rows found. Check dataset file.")
        sys.exit(1)

    X = np.array(rows_X, dtype=np.float32)
    Y = np.array(rows_Y, dtype=np.int32)

    print("[DATA] Rows loaded     : " + str(len(rows_X)))
    print("[DATA] X shape         : " + str(X.shape))
    print("[DATA] Y shape         : " + str(Y.shape))
    print("[DATA] Resource classes: " + str(sorted(np.unique(Y))))

    return X, Y


def split_and_normalize(X, Y):
    """
    Splits data into 80/20 train/test sets and applies StandardScaler.
    Scaler is fit on training set only and saved to disk.
    evaluate.py and figures.py load the saved scaler automatically.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("[DATA] Train samples   : " + str(len(X_train)))
    print("[DATA] Test samples    : " + str(len(X_test)))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_FILE)
    print("[DATA] Scaler saved    : " + SCALER_FILE)

    return X_train, X_test, Y_train, Y_test


def load_dataset(fname):
    """
    Full pipeline: parse, split, normalize.
    Returns X_train, X_test, Y_train, Y_test.
    """
    X, Y = parse_dataset(fname)
    return split_and_normalize(X, Y)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py fogmls_dataset_seed42.txt")
        sys.exit(1)
    X_train, X_test, Y_train, Y_test = load_dataset(sys.argv[1])
    print("[DATA] Ready.")
