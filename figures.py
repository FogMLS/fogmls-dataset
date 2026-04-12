# -*- coding: utf-8 -*-
"""
figures.py - FogMLS Figure Generator
Generates four SVG figures for the dataset paper:
  1. confusion_matrix.svg
  2. loss_curve.svg
  3. roc_curve.svg
  4. f1_bar_chart.svg

Usage:
    python figures.py fogmls_dataset_seed42.txt

Requires in same folder:
    DL_weights_200.weights.h5   (produced by train.py)
    DL_scaler_200.joblib        (produced by train.py)
    training_history.json       (produced by train.py)

Dataset download:
    https://doi.org/10.5281/zenodo.19382240
"""

import sys
import os
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             f1_score, classification_report)
from sklearn.preprocessing import label_binarize

from data_loader import load_dataset

# ================================================================
# CONSTANTS
# ================================================================
MAX_JOBS         = 200
MAX_RS           = 5
FEATURES_PER_JOB = 3
WEIGHTS_FILE     = 'DL_weights_200.weights.h5'
SCALER_FILE      = 'DL_scaler_200.joblib'
HISTORY_FILE     = 'training_history.json'

CLASS_NAMES  = ['Nothing', 'Cloud', 'Fog-1', 'Fog-2', 'Fog-3']
CLASS_COLORS = ['#AAAAAA', '#4E9AF1', '#F4A261', '#2EC4B6', '#E76F51']


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
# GET PREDICTIONS
# ================================================================
def get_predictions(X_test, Y_test):
    model, _ = build_model()
    model.load_weights(WEIGHTS_FILE)
    print("[FIGURES] Weights loaded: " + WEIGHTS_FILE)

    predictions = model.predict(X_test, verbose=0)

    all_true, all_pred, all_prob = [], [], []
    for job in range(MAX_JOBS):
        pred_labels = np.argmax(predictions[job], axis=1)
        true_labels = np.clip(Y_test[:, job], 0, MAX_RS - 1)
        active_mask = true_labels > 0
        if active_mask.sum() > 0:
            all_true.extend(true_labels[active_mask].tolist())
            all_pred.extend(pred_labels[active_mask].tolist())
            all_prob.extend(predictions[job][active_mask].tolist())

    return (np.array(all_true),
            np.array(all_pred),
            np.array(all_prob))


# ================================================================
# FIGURE 1 - Confusion Matrix
# ================================================================
def plot_confusion_matrix(all_true, all_pred):
    print("[FIGURES] Generating confusion_matrix.svg ...")

    labels     = list(range(1, MAX_RS))
    tick_names = CLASS_NAMES[1:]

    cm      = confusion_matrix(all_true, all_pred, labels=labels)
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation='nearest',
                   cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_names, fontsize=10)
    ax.set_yticklabels(tick_names, fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=11)
    ax.set_ylabel('True label', fontsize=11)
    ax.set_title('Confusion Matrix (normalised by row)', fontsize=12)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = 'white' if cm_norm[i, j] > thresh else 'black'
            ax.text(j, i,
                    str(cm[i, j]) + '\n(' +
                    '{:.2f}'.format(cm_norm[i, j]) + ')',
                    ha='center', va='center',
                    color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig('confusion_matrix.svg', format='svg', bbox_inches='tight')
    plt.close()
    print("[FIGURES] Saved: confusion_matrix.svg")


# ================================================================
# FIGURE 2 - Loss Curve
# ================================================================
def plot_loss_curve():
    print("[FIGURES] Generating loss_curve.svg ...")

    if not os.path.exists(HISTORY_FILE):
        print("[FIGURES] " + HISTORY_FILE +
              " not found. Skipping loss curve.")
        return

    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)

    train_keys = [k for k in history if not k.startswith('val_')]
    val_keys   = [k for k in history if k.startswith('val_')]
    epochs     = range(1, len(history[train_keys[0]]) + 1)

    train_loss = np.mean([history[k] for k in train_keys], axis=0)
    val_loss   = np.mean([history[k] for k in val_keys],   axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, color='#4E9AF1',
            linewidth=1.8, label='Training loss')
    ax.plot(epochs, val_loss, color='#E76F51',
            linewidth=1.8, linestyle='--', label='Validation loss')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Mean categorical cross-entropy loss', fontsize=11)
    ax.set_title('Training and Validation Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))

    plt.tight_layout()
    plt.savefig('loss_curve.svg', format='svg', bbox_inches='tight')
    plt.close()
    print("[FIGURES] Saved: loss_curve.svg")


# ================================================================
# FIGURE 3 - ROC Curve
# ================================================================
def plot_roc_curve(all_true, all_prob):
    print("[FIGURES] Generating roc_curve.svg ...")

    labels = list(range(1, MAX_RS))
    y_bin  = label_binarize(all_true, classes=list(range(MAX_RS)))

    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, label], all_prob[:, label])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                color=CLASS_COLORS[label],
                linewidth=1.8,
                label=CLASS_NAMES[label] +
                      ' (AUC = ' + '{:.3f}'.format(roc_auc) + ')')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False positive rate', fontsize=11)
    ax.set_ylabel('True positive rate', fontsize=11)
    ax.set_title('ROC Curves (one-vs-rest)', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig('roc_curve.svg', format='svg', bbox_inches='tight')
    plt.close()
    print("[FIGURES] Saved: roc_curve.svg")


# ================================================================
# FIGURE 4 - F1 Bar Chart
# ================================================================
def plot_f1_bar_chart(all_true, all_pred):
    print("[FIGURES] Generating f1_bar_chart.svg ...")

    labels     = list(range(1, MAX_RS))
    tick_names = CLASS_NAMES[1:]

    f1_scores = f1_score(all_true, all_pred,
                         labels=labels,
                         average=None,
                         zero_division=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(tick_names, f1_scores,
                  color=CLASS_COLORS[1:],
                  edgecolor='black',
                  linewidth=0.6)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                '{:.3f}'.format(score),
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Resource class', fontsize=11)
    ax.set_ylabel('F1 score', fontsize=11)
    ax.set_title('Per-class F1 Score', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('f1_bar_chart.svg', format='svg', bbox_inches='tight')
    plt.close()
    print("[FIGURES] Saved: f1_bar_chart.svg")


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python figures.py fogmls_dataset_seed42.txt")
        sys.exit(1)

    for f in [WEIGHTS_FILE, SCALER_FILE]:
        if not os.path.exists(f):
            print("[FIGURES] Required file not found: " + f)
            print("[FIGURES] Run train.py first.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  FOGMLS - FIGURE GENERATOR")
    print("=" * 60 + "\n")

    _, X_test, _, Y_test = load_dataset(sys.argv[1])
    all_true, all_pred, all_prob = get_predictions(X_test, Y_test)

    plot_confusion_matrix(all_true, all_pred)
    plot_loss_curve()
    plot_roc_curve(all_true, all_prob)
    plot_f1_bar_chart(all_true, all_pred)

    print("\n[FIGURES] All figures saved.")
    print("[FIGURES] Open SVG files in Inkscape for publication export.")
    print("=" * 60)
