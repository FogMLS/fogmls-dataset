# -*- coding: utf-8 -*-
"""
evaluate.py - FogMLS Model Evaluation Script
Loads trained weights and produces per-class F1, precision,
recall, and accuracy results reported in the dataset paper.

Usage:
    python evaluate.py fogmls_dataset_seed42.txt

Requires in same folder:
    DL_weights_200.weights.h5   (produced by train.py)
    DL_scaler_200.joblib        (produced by train.py)

Dataset download:
    https://doi.org/10.5281/zenodo.19382240
"""

import sys
import os
import numpy as np
import joblib
import keras
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from sklearn.metrics import f1_score, classification_report

from data_loader import load_dataset

# ================================================================
# CONSTANTS
# ================================================================
MAX_JOBS         = 200
MAX_RS           = 5
FEATURES_PER_JOB = 3
WEIGHTS_FILE     = 'DL_weights_200.weights.h5'
SCALER_FILE      = 'DL_scaler_200.joblib'

CLASS_NAMES = ['Nothing', 'Cloud', 'Fog-1', 'Fog-2', 'Fog-3']


# ================================================================
# MODEL ARCHITECTURE
# Must match train.py exactly.
# ================================================================
def build_model():
    inputs = keras.Input(shape=(MAX_JOBS * FEATURES_PER_JOB,))
    shared = Dense(128, activation='relu')(inputs)

    outputs      = []
    output_names = []
    for job in range(MAX_JOBS):
        name = 'job_' + str(job)
        head = Dense(64, activation='relu')(shared)
        out  = Dense(MAX_RS, activation='softmax', name=name)(head)
        outputs.append(out)
        output_names.append(name)

    model = keras.Model(inputs=inputs, outputs=outputs)
    loss  = {name: 'categorical_crossentropy' for name in output_names}
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss)
    return model, output_names


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py fogmls_dataset_seed42.txt")
        sys.exit(1)

    for f in [WEIGHTS_FILE, SCALER_FILE]:
        if not os.path.exists(f):
            print("[EVAL] Required file not found: " + f)
            print("[EVAL] Run train.py first.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  FOGMLS - MODEL EVALUATION")
    print("=" * 60 + "\n")

    _, X_test, _, Y_test = load_dataset(sys.argv[1])

    model, _ = build_model()
    model.load_weights(WEIGHTS_FILE)
    print("[EVAL] Weights loaded: " + WEIGHTS_FILE)

    print("[EVAL] Running predictions...")
    predictions = model.predict(X_test, verbose=0)

    all_true, all_pred = [], []
    job_accs = []

    for job in range(MAX_JOBS):
        pred_labels = np.argmax(predictions[job], axis=1)
        true_labels = np.clip(Y_test[:, job], 0, MAX_RS - 1)
        active_mask = true_labels > 0
        if active_mask.sum() > 0:
            acc = np.mean(pred_labels[active_mask] ==
                          true_labels[active_mask])
            job_accs.append(acc)
            all_true.extend(true_labels[active_mask].tolist())
            all_pred.extend(pred_labels[active_mask].tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    overall_acc = np.mean(all_pred == all_true)
    macro_f1    = f1_score(all_true, all_pred, average='macro',
                           labels=list(range(MAX_RS)), zero_division=0)
    weighted_f1 = f1_score(all_true, all_pred, average='weighted',
                           labels=list(range(MAX_RS)), zero_division=0)

    report = classification_report(
        all_true, all_pred,
        labels=list(range(MAX_RS)),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print("  Test samples          : " + str(len(X_test)))
    print("  Overall accuracy      : " +
          str(round(overall_acc * 100, 2)) + "%")
    print("  Macro F1              : " + str(round(macro_f1, 4)))
    print("  Weighted F1           : " + str(round(weighted_f1, 4)))
    print("  Mean per-job accuracy : " +
          str(round(np.mean(job_accs) * 100, 2)) + "%")
    print("  Min per-job accuracy  : " +
          str(round(np.min(job_accs) * 100, 2)) + "%")
    print("  Max per-job accuracy  : " +
          str(round(np.max(job_accs) * 100, 2)) + "%")
    print("-" * 60)
    print("  Per-class breakdown:")
    print(report)
    print("=" * 60)
    print("\n[EVAL] Done. Run figures.py to generate figures.")
