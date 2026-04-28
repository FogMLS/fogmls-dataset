# -*- coding: utf-8 -*-
"""
DL_PIPELINE.py - FogMLS End-to-End DL Training Pipeline
Usage:
    python3 DL_PIPELINE.py                          # auto-find latest dataset
    python3 DL_PIPELINE.py dataset-filename.txt     # use specific dataset file
"""

import sys
import os
import glob
import numpy as np
import joblib
from keras.utils import to_categorical
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from config import (
    MAX_JOBS, MAX_RS, FEATURES_PER_JOB, NUM_FOG_NODES
)
from model_arch import build_model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("[PIPELINE] Running on GPU: " + str(gpus[0].name))
else:
    print("[PIPELINE] No GPU found - running on CPU")

EPOCHS = 100
BATCH_SIZE = 128
RANDOM_STATE = 42

WEIGHTS_FILE = 'DL_weights_' + str(MAX_JOBS) + '.weights.h5'
SCALER_FILE = 'DL_scaler_' + str(MAX_JOBS) + '.joblib'


def get_class_name(cls):
    """Returns human readable class name for a given label integer."""
    if int(cls) == 0:
        return 'Nothing'
    elif int(cls) == 1:
        return 'Cloud'
    else:
        return 'Fog-' + str(int(cls) - 1)


# STEP 1 - Find dataset file
def find_dataset_file():
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        if not os.path.exists(fname):
            print("[ERROR] File not found: " + fname)
            sys.exit(1)
        print("[PIPELINE] Using specified file: " + fname)
        return fname
    else:
        files = (glob.glob('dataset-*.txt') +
                 glob.glob('fogmls_dataset_*.txt'))
        if not files:
            print("[ERROR] No dataset files found. " +
                  "Run IoTsimulation.py with GA_RM first.")
            sys.exit(1)
        latest = max(files, key=os.path.getmtime)
        print("[PIPELINE] Auto-found latest dataset: " + latest)
        return latest


# STEP 2 - Parse dataset
def parse_dataset(fname):
    print("[PIPELINE] Parsing dataset: " + fname)
    rows_X = []
    rows_Y = []
    skipped_rows = 0
    with open(fname, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            skipped_rows += 1
            continue
        if any(c.isalpha() for c in line):
            skipped_rows += 1
            continue
        try:
            values = [float(v.strip()) for v in line.split(',')
                      if v.strip() != '']
        except ValueError:
            skipped_rows += 1
            continue
        expected = MAX_JOBS * FEATURES_PER_JOB + MAX_JOBS
        if len(values) < expected:
            skipped_rows += 1
            continue
        x_row = values[:MAX_JOBS * FEATURES_PER_JOB]
        y_row = values[MAX_JOBS * FEATURES_PER_JOB:
                       MAX_JOBS * FEATURES_PER_JOB + MAX_JOBS]
        rows_X.append(x_row)
        rows_Y.append(y_row)
    if skipped_rows > 0:
        print("[PIPELINE] Skipped " + str(skipped_rows) +
              " non-data rows (header, empty, malformed)")
    if len(rows_X) == 0:
        print("[ERROR] No valid data rows found. " +
              "Check simulation output.")
        sys.exit(1)
    X = np.array(rows_X, dtype=np.float32)
    Y = np.array(rows_Y, dtype=np.int32)

    # Validate label range
    y_min = int(Y.min())
    y_max = int(Y.max())
    if y_min < 0 or y_max >= MAX_RS:
        print("[ERROR] Labels out of range [0, " + str(MAX_RS - 1) +
              "]. Found min=" + str(y_min) + " max=" + str(y_max))
        print("[ERROR] This indicates a device-to-class mapping problem.")
        sys.exit(1)

    print("[PIPELINE] Parsed " + str(len(rows_X)) + " slots (rows)")
    print("[PIPELINE] X shape: " + str(X.shape))
    print("[PIPELINE] Y shape: " + str(Y.shape))
    print("[PIPELINE] Resource classes found: " +
          str(sorted(np.unique(Y))))
    return X, Y


# STEP 3 - Train/test split
def split_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_STATE
    )
    print("[PIPELINE] Train samples: " + str(len(X_train)) +
          "  Test samples: " + str(len(X_test)))
    return X_train, X_test, Y_train, Y_test


# STEP 3b - Print training set class distribution
def print_class_distribution(Y_train):
    print("[PIPELINE] Training set class distribution:")
    unique, counts = np.unique(Y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print("  Class " + str(cls) +
              " (" + get_class_name(cls) + "): " + str(cnt))


# STEP 3c - Normalize features
def normalize_features(X_train, X_test):
    """
    Applies StandardScaler normalization to feature columns.
    Capacity feature values range from 0 to SLOT_TIME (35 SU)
    while task type values range from 0 to 5. Without normalization,
    capacity features can dominate the model input.
    Scaler is fit on training set only and applied to both sets.
    Scaler is saved to disk so IoTsimulation.py can apply the same
    normalization during AI scheduler deployment.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_FILE)
    print("[PIPELINE] Features normalized. Scaler saved: " + SCALER_FILE)
    return X_train_scaled, X_test_scaled, scaler


# STEP 5 - Prepare one-hot labels
def prepare_labels(Y):
    Z = []
    for job in range(MAX_JOBS):
        labels = np.clip(Y[:, job], 0, MAX_RS - 1)
        Z.append(to_categorical(labels, num_classes=MAX_RS))
    return Z


# STEP 5b - Find latest epoch checkpoint for resume
def find_latest_checkpoint():
    checkpoints = glob.glob('checkpoint_epoch_*.keras')
    if not checkpoints:
        return None, 0
    latest = max(checkpoints, key=os.path.getmtime)
    try:
        epoch_num = int(
            latest.replace('checkpoint_epoch_', '').replace('.keras', '')
        )
    except ValueError:
        return None, 0
    print("[PIPELINE] Found checkpoint at epoch " +
          str(epoch_num) + ": " + latest)
    return latest, epoch_num


# STEP 5c - Compute per-class weights for each output head
def compute_class_weights(Y_train):
    """
    Computes class weights for the loss function, applied per output
    head. This operates at the correct granularity: per-class within
    each task position, not per-sample.

    Fog node labels (2 to MAX_RS-1) are weighted inversely
    proportional to their frequency. Cloud (1) gets weight 1.0.
    Nothing (0) gets weight near zero since it is just padding.

    Returns a dict of {output_name: {class_index: weight}}.
    """
    # Compute global class frequencies across all active positions
    active_labels = Y_train[Y_train > 0]
    if len(active_labels) == 0:
        print("[PIPELINE] No active labels found. Using uniform weights.")
        return None

    unique, counts = np.unique(active_labels, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))

    # Find the majority active class count for normalization
    majority_count = max(class_counts.values())

    class_weight_dict = {}
    # Nothing class gets minimal weight (padding, not real decisions)
    class_weight_dict[0] = 0.01

    for cls in range(1, MAX_RS):
        cnt = class_counts.get(cls, 0)
        if cnt == 0:
            # Class is absent from this scenario (e.g. Cloud in low load).
            # Weight set to 0.0 so it does not distort the loss function.
            class_weight_dict[cls] = 0.0
        else:
            class_weight_dict[cls] = majority_count / cnt

    print("[PIPELINE] Per-class weights for loss:")
    for cls, w in class_weight_dict.items():
        print("  Class " + str(cls) +
              " (" + get_class_name(cls) + "): " +
              str(round(w, 4)))

    return class_weight_dict


# STEP 5d - Cleanup epoch checkpoints after successful run
def cleanup_checkpoints():
    checkpoints = glob.glob('checkpoint_epoch_*.keras')
    for f in checkpoints:
        try:
            os.remove(f)
        except OSError:
            pass
    if checkpoints:
        print("[PIPELINE] Cleaned up " + str(len(checkpoints)) +
              " checkpoint files.")


# STEP 6 - Train model
def train_model(model, X_train, Z_train, initial_epoch=0):
    import keras
    print("[PIPELINE] Training for " + str(EPOCHS) + " epochs...")
    if initial_epoch > 0:
        print("[PIPELINE] Resuming from epoch " + str(initial_epoch))
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            verbose=0
        ),
        keras.callbacks.ModelCheckpoint(
            'checkpoint_epoch_{epoch:03d}.keras',
            save_freq='epoch',
            verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    # Class weights are baked into the compiled loss inside build_model().
    # model.fit() does not receive class_weight because Keras only supports
    # that argument for single-output models.
    history = model.fit(
        X_train,
        Z_train,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    import json
    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f)
    print("[PIPELINE] Training history saved: training_history.json")
    print("[PIPELINE] Training complete.")
    return history


# STEP 7 - Evaluate model
def evaluate_model(model, X_test, Y_test, Z_test):
    print("\n[PIPELINE] Evaluating on test set...")
    predictions = model.predict(X_test, verbose=0)
    all_true = []
    all_pred = []
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
    macro_f1 = f1_score(all_true, all_pred, average='macro',
                        labels=list(range(MAX_RS)),
                        zero_division=0)
    weighted_f1 = f1_score(all_true, all_pred, average='weighted',
                           labels=list(range(MAX_RS)),
                           zero_division=0)

    # Dynamic target names
    target_names = (['Nothing', 'Cloud'] +
                    ['Fog-' + str(i) for i in range(1, MAX_RS - 1)])

    report = classification_report(
        all_true, all_pred,
        labels=list(range(MAX_RS)),
        target_names=target_names,
        zero_division=0
    )
    print("\n" + "=" * 60)
    print("  FOGMLS DL PIPELINE - RESULTS TABLE")
    print("=" * 60)
    print("  Test samples          : " + str(len(X_test)))
    print("  Overall accuracy      : " +
          str(round(overall_acc * 100, 2)) + "%")
    print("  Macro F1 score        : " + str(round(macro_f1, 4)))
    print("  Weighted F1 score     : " + str(round(weighted_f1, 4)))
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
    return overall_acc, macro_f1


# STEP 8 - Save weights
def save_weights(model):
    model.save_weights(WEIGHTS_FILE)
    print("[PIPELINE] Weights saved: " + WEIGHTS_FILE)


# MAIN
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  FOGMLS DL PIPELINE STARTING")
    print("=" * 60 + "\n")

    dataset_file = find_dataset_file()
    X, Y = parse_dataset(dataset_file)

    if len(X) < 5:
        print("[WARNING] Only " + str(len(X)) +
              " slots found. Run simulation longer.")

    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # Normalize features before training
    X_train, X_test, scaler = normalize_features(X_train, X_test)

    # Print real training distribution before any weighting
    print_class_distribution(Y_train)

    # Class weights are passed into build_model so they are compiled
    # into the loss function, which works for multi-output models.
    class_weight_dict = compute_class_weights(Y_train)
    model, output_names = build_model(class_weight_dict)

    Z_train = prepare_labels(Y_train)
    Z_test = prepare_labels(Y_test)

    # Check for existing checkpoint to resume from crash
    checkpoint_path, initial_epoch = find_latest_checkpoint()
    if checkpoint_path:
        print("[PIPELINE] Loading checkpoint: " + checkpoint_path)
        model.load_weights(checkpoint_path)
    else:
        print("[PIPELINE] No checkpoint found. Starting fresh.")

    history = train_model(model, X_train, Z_train, initial_epoch)

    overall_acc, macro_f1 = evaluate_model(
        model, X_test, Y_test, Z_test)

    save_weights(model)

    cleanup_checkpoints()

    print("\n[PIPELINE] Done.")
    print("[PIPELINE] Set RM_TYPE = AI_RM in config.py " +
          "to use DL scheduler.")
    print("[PIPELINE] Ensure " + WEIGHTS_FILE +
          " and " + SCALER_FILE +
          " are in the same folder as IoTsimulation.py\n")