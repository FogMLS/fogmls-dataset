# -*- coding: utf-8 -*-
"""
train.py - FogMLS Model Training Script
Trains a multi-output deep learning model on the FogMLS dataset.

Usage:
    python train.py fogmls_dataset_seed42.txt

Saves:
    DL_weights_200.weights.h5   - trained model weights
    DL_scaler_200.joblib        - fitted StandardScaler
    training_history.json       - loss curves per epoch

Dataset download:
    https://doi.org/10.5281/zenodo.19382240
"""

import sys
import os
import json
import glob
import numpy as np
import tensorflow as tf
import keras
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from keras.utils import to_categorical

from data_loader import load_dataset

# ================================================================
# CONSTANTS
# ================================================================
MAX_JOBS         = 200
MAX_RS           = 5
FEATURES_PER_JOB = 3

EPOCHS           = 100
BATCH_SIZE       = 128
RANDOM_STATE     = 42
WEIGHTS_FILE     = 'DL_weights_200.weights.h5'

# ================================================================
# GPU CHECK
# ================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("[TRAIN] Running on GPU: " + str(gpus[0].name))
else:
    print("[TRAIN] No GPU found - running on CPU")


def get_class_name(cls):
    if int(cls) == 0:
        return 'Nothing'
    elif int(cls) == 1:
        return 'Cloud'
    return 'Fog-' + str(int(cls) - 1)


# ================================================================
# MODEL ARCHITECTURE
# Multi-output MLP: shared Dense(128) backbone with one
# Dense(64) head per task position, each outputting softmax
# over MAX_RS resource classes.
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

    print("[TRAIN] Model built: Input=" + str(MAX_JOBS * FEATURES_PER_JOB) +
          ", Jobs=" + str(MAX_JOBS) +
          ", Classes=" + str(MAX_RS))
    return model, output_names


# ================================================================
# INVERSE FREQUENCY WEIGHTING
# Weights training samples by inverse fog class frequency.
# Nothing (0) and Cloud (1) are excluded from reweighting.
# ================================================================
def compute_sample_weights(Y_train):
    unique, counts = np.unique(Y_train, return_counts=True)
    class_counts   = dict(zip(unique.tolist(), counts.tolist()))

    fog_labels  = list(range(2, MAX_RS))
    fog_counts  = {cls: class_counts.get(cls, 1) for cls in fog_labels}

    if not fog_counts:
        print("[TRAIN] No fog classes found. Using uniform weights.")
        return np.ones(len(Y_train), dtype=np.float32)

    majority = max(fog_counts.values())
    fog_weights = {cls: majority / cnt for cls, cnt in fog_counts.items()}

    print("[TRAIN] Inverse frequency weights:")
    for cls, w in fog_weights.items():
        print("  " + get_class_name(cls) + ": " + str(round(w, 4)))

    weights = np.ones(len(Y_train), dtype=np.float32)
    for i in range(len(Y_train)):
        row = Y_train[i]
        max_w = 1.0
        for fog_lbl, w in fog_weights.items():
            if np.any(row == fog_lbl) and w > max_w:
                max_w = w
        weights[i] = max_w

    return weights


# ================================================================
# CHECKPOINT RESUME
# ================================================================
def find_latest_checkpoint():
    checkpoints = glob.glob('checkpoint_epoch_*.keras')
    if not checkpoints:
        return None, 0
    latest    = max(checkpoints, key=os.path.getmtime)
    try:
        epoch_num = int(
            latest.replace('checkpoint_epoch_', '').replace('.keras', ''))
    except ValueError:
        return None, 0
    print("[TRAIN] Resuming from epoch " + str(epoch_num) + ": " + latest)
    return latest, epoch_num


def cleanup_checkpoints():
    checkpoints = glob.glob('checkpoint_epoch_*.keras')
    for f in checkpoints:
        try:
            os.remove(f)
        except OSError:
            pass
    if checkpoints:
        print("[TRAIN] Cleaned up " + str(len(checkpoints)) +
              " checkpoint files.")


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train.py fogmls_dataset_seed42.txt")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  FOGMLS - MODEL TRAINING")
    print("=" * 60 + "\n")

    X_train, X_test, Y_train, Y_test = load_dataset(sys.argv[1])

    print("[TRAIN] Class distribution in training set:")
    unique, counts = np.unique(Y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print("  " + get_class_name(cls) + ": " + str(cnt))

    model, output_names = build_model()

    Z_train = [to_categorical(
                   np.clip(Y_train[:, j], 0, MAX_RS - 1),
                   num_classes=MAX_RS)
               for j in range(MAX_JOBS)]

    sample_weights = compute_sample_weights(Y_train)

    checkpoint_path, initial_epoch = find_latest_checkpoint()
    if checkpoint_path:
        model.load_weights(checkpoint_path)
    else:
        print("[TRAIN] No checkpoint found. Starting fresh.")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_model.keras', save_best_only=True, verbose=0),
        keras.callbacks.ModelCheckpoint(
            'checkpoint_epoch_{epoch:03d}.keras',
            save_freq='epoch', verbose=0),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1),
    ]

    print("[TRAIN] Training for " + str(EPOCHS) + " epochs...")
    history = model.fit(
        X_train, Z_train,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        sample_weight=sample_weights,
        verbose=1
    )

    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f)
    print("[TRAIN] History saved: training_history.json")

    model.save_weights(WEIGHTS_FILE)
    print("[TRAIN] Weights saved: " + WEIGHTS_FILE)

    cleanup_checkpoints()

    print("\n[TRAIN] Done. Run evaluate.py next.")
    print("=" * 60)
