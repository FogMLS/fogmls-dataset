"""
comprehensive_validation.py
============================
Comprehensive validation of FogMLS dataset and DL model.
Produces results suitable for journal supplementary material.

Checks performed:
1. Decision tree baseline (depth 3, 5, 10) vs DL model
2. Random forest baseline vs DL model
3. Formal confusion matrix with per-class analysis
4. ROC curves (one-vs-rest) per class
5. Precision-recall curves per class
6. Per-class F1, precision, recall breakdown
7. Summary verdict

Usage:
    python3 comprehensive_validation.py <dataset_file>

Example:
    python3 comprehensive_validation.py dataset-21-04-202618-56-47.txt
"""

import sys
import os
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

from config import MAX_JOBS, MAX_RS, FEATURES_PER_JOB
from model_arch import build_model

WEIGHTS_FILE = 'DL_weights_' + str(MAX_JOBS) + '.weights.h5'
SCALER_FILE  = 'DL_scaler_'  + str(MAX_JOBS) + '.joblib'
RANDOM_STATE = 42
ACTIVE_CLASSES = list(range(1, MAX_RS))  # exclude Nothing class
CLASS_NAMES = {0: 'Nothing', 1: 'Cloud', 2: 'Fog-1', 3: 'Fog-2', 4: 'Fog-3'}


# ================================================================
# DATA LOADING
# ================================================================

def parse(fname):
    rows_X, rows_Y, skipped = [], [], 0
    expected = MAX_JOBS * FEATURES_PER_JOB + MAX_JOBS
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or any(c.isalpha() for c in line):
                skipped += 1
                continue
            try:
                vals = [float(v) for v in line.split(',') if v.strip()]
            except ValueError:
                skipped += 1
                continue
            if len(vals) < expected:
                skipped += 1
                continue
            rows_X.append(vals[:MAX_JOBS * FEATURES_PER_JOB])
            rows_Y.append(vals[MAX_JOBS * FEATURES_PER_JOB:
                               MAX_JOBS * FEATURES_PER_JOB + MAX_JOBS])
    print('[PARSE] {} rows, {} skipped'.format(len(rows_X), skipped))
    return np.array(rows_X, dtype=np.float32), np.array(rows_Y, dtype=np.int32)


def get_active_flat(X, Y, scaler=None):
    """Flatten all active task positions into per-task feature/label pairs."""
    flat_X, flat_Y = [], []
    for row_idx in range(len(X)):
        for job in range(MAX_JOBS):
            label = Y[row_idx, job]
            if label == 0:
                continue
            base = job * FEATURES_PER_JOB
            features = X[row_idx, base:base + FEATURES_PER_JOB]
            flat_X.append(features)
            flat_Y.append(label)
    flat_X = np.array(flat_X, dtype=np.float32)
    flat_Y = np.array(flat_Y, dtype=np.int32)
    if scaler is not None:
        # Create full-size input for scaler then extract position features
        # Use per-task features directly since scaler was fit on slot-level
        # We refit a simple scaler on the flat features for baseline models
        pass
    return flat_X, flat_Y


def get_dl_predictions(model, X_test, Y_test, scaler):
    """Get DL model predictions on test set."""
    X_scaled = scaler.transform(X_test)
    predictions = model.predict(X_scaled, verbose=0)

    all_true, all_pred, all_proba = [], [], []
    for job in range(MAX_JOBS):
        pred_proba = predictions[job]
        pred_labels = np.argmax(pred_proba, axis=1)
        true_labels = np.clip(Y_test[:, job], 0, MAX_RS - 1)
        active_mask = true_labels > 0
        if active_mask.sum() > 0:
            all_true.extend(true_labels[active_mask].tolist())
            all_pred.extend(pred_labels[active_mask].tolist())
            all_proba.extend(pred_proba[active_mask].tolist())

    return (np.array(all_true), np.array(all_pred),
            np.array(all_proba))


# ================================================================
# CHECK 1: DECISION TREE AND RANDOM FOREST BASELINES
# ================================================================

def check_baselines(flat_X_train, flat_Y_train, flat_X_test, flat_Y_test,
                    results_file):
    print('\n' + '=' * 60)
    print('CHECK 1: Baseline Models vs DL')
    print('=' * 60)

    # Normalise flat features for baselines
    from sklearn.preprocessing import StandardScaler as SS
    sc = SS()
    flat_X_train_sc = sc.fit_transform(flat_X_train)
    flat_X_test_sc = sc.transform(flat_X_test)

    baselines = {}

    # Decision trees at different depths
    for depth in [3, 5, 10]:
        dt = DecisionTreeClassifier(
            max_depth=depth, random_state=RANDOM_STATE)
        dt.fit(flat_X_train_sc, flat_Y_train)
        pred = dt.predict(flat_X_test_sc)
        acc = np.mean(pred == flat_Y_test) * 100
        f1 = f1_score(flat_Y_test, pred, average='macro',
                      labels=ACTIVE_CLASSES, zero_division=0)
        name = 'Decision Tree (depth={})'.format(depth)
        baselines[name] = {'accuracy': acc, 'macro_f1': f1}
        print('  {}: accuracy={:.2f}%, macro_f1={:.4f}'.format(
            name, acc, f1))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(flat_X_train_sc, flat_Y_train)
    pred_rf = rf.predict(flat_X_test_sc)
    acc_rf = np.mean(pred_rf == flat_Y_test) * 100
    f1_rf = f1_score(flat_Y_test, pred_rf, average='macro',
                     labels=ACTIVE_CLASSES, zero_division=0)
    baselines['Random Forest (100 trees)'] = {
        'accuracy': acc_rf, 'macro_f1': f1_rf}
    print('  Random Forest (100 trees): accuracy={:.2f}%, macro_f1={:.4f}'.format(
        acc_rf, f1_rf))

    with open(results_file, 'a') as f:
        f.write('\n' + '=' * 60 + '\n')
        f.write('CHECK 1: Baseline Models\n')
        f.write('=' * 60 + '\n')
        for name, res in baselines.items():
            f.write('  {}: accuracy={:.2f}%, macro_f1={:.4f}\n'.format(
                name, res['accuracy'], res['macro_f1']))

    return baselines


# ================================================================
# CHECK 2: CONFUSION MATRIX
# ================================================================

def check_confusion_matrix(true, pred, results_file, label):
    print('\n' + '=' * 60)
    print('CHECK 2: Confusion Matrix ({})'.format(label))
    print('=' * 60)

    active_classes = [c for c in ACTIVE_CLASSES if c in np.unique(true)]
    class_names = [CLASS_NAMES[c] for c in active_classes]

    cm = confusion_matrix(true, pred, labels=active_classes)

    # Print confusion matrix
    header = '{:>10}'.format('') + ''.join(
        '{:>10}'.format(n) for n in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row = '{:>10}'.format(row_name) + ''.join(
            '{:>10}'.format(cm[i, j]) for j in range(len(active_classes)))
        print(row)

    # Find most common errors
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    total_errors = cm_no_diag.sum()
    print('\n  Total errors: {}'.format(total_errors))
    print('  Error rate: {:.4f}%'.format(
        100 * total_errors / cm.sum()))

    if total_errors > 0:
        err_row, err_col = np.unravel_index(
            cm_no_diag.argmax(), cm_no_diag.shape)
        print('  Most common error: true={} predicted as {} ({} times)'.format(
            class_names[err_row], class_names[err_col],
            cm_no_diag[err_row, err_col]))

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix - {}'.format(label))
    plt.tight_layout()
    plot_file = 'confusion_matrix_{}.png'.format(
        label.lower().replace(' ', '_').replace('(', '').replace(')', ''))
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print('  Confusion matrix saved: {}'.format(plot_file))

    with open(results_file, 'a') as f:
        f.write('\n' + '=' * 60 + '\n')
        f.write('CHECK 2: Confusion Matrix ({})\n'.format(label))
        f.write('=' * 60 + '\n')
        f.write(header + '\n')
        for i, row_name in enumerate(class_names):
            row = '{:>10}'.format(row_name) + ''.join(
                '{:>10}'.format(cm[i, j]) for j in range(len(active_classes)))
            f.write(row + '\n')
        f.write('\nTotal errors: {}\n'.format(total_errors))
        f.write('Error rate: {:.4f}%\n'.format(
            100 * total_errors / cm.sum()))

    return cm, total_errors


# ================================================================
# CHECK 3: ROC CURVES
# ================================================================

def check_roc(true, proba, results_file, label):
    print('\n' + '=' * 60)
    print('CHECK 3: ROC Curves ({})'.format(label))
    print('=' * 60)

    active_classes = [c for c in ACTIVE_CLASSES if c in np.unique(true)]

    # Binarize labels for one-vs-rest ROC
    true_bin = label_binarize(true, classes=list(range(MAX_RS)))

    fig, ax = plt.subplots(figsize=(8, 6))
    roc_results = {}

    for cls in active_classes:
        if cls >= proba.shape[1]:
            continue
        fpr, tpr, _ = roc_curve(true_bin[:, cls], proba[:, cls])
        roc_auc = auc(fpr, tpr)
        roc_results[CLASS_NAMES[cls]] = roc_auc
        ax.plot(fpr, tpr, label='{} (AUC={:.4f})'.format(
            CLASS_NAMES[cls], roc_auc))
        print('  {}: AUC={:.4f}'.format(CLASS_NAMES[cls], roc_auc))

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - {}'.format(label))
    ax.legend(loc='lower right')
    plt.tight_layout()
    plot_file = 'roc_curves_{}.png'.format(
        label.lower().replace(' ', '_').replace('(', '').replace(')', ''))
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print('  ROC curves saved: {}'.format(plot_file))

    with open(results_file, 'a') as f:
        f.write('\n' + '=' * 60 + '\n')
        f.write('CHECK 3: ROC AUC Scores ({})\n'.format(label))
        f.write('=' * 60 + '\n')
        for cls_name, roc_auc in roc_results.items():
            f.write('  {}: AUC={:.4f}\n'.format(cls_name, roc_auc))

    return roc_results


# ================================================================
# CHECK 4: PRECISION-RECALL CURVES
# ================================================================

def check_precision_recall(true, proba, results_file, label):
    print('\n' + '=' * 60)
    print('CHECK 4: Precision-Recall Curves ({})'.format(label))
    print('=' * 60)

    active_classes = [c for c in ACTIVE_CLASSES if c in np.unique(true)]
    true_bin = label_binarize(true, classes=list(range(MAX_RS)))

    fig, ax = plt.subplots(figsize=(8, 6))
    pr_results = {}

    for cls in active_classes:
        if cls >= proba.shape[1]:
            continue
        precision, recall, _ = precision_recall_curve(
            true_bin[:, cls], proba[:, cls])
        ap = average_precision_score(true_bin[:, cls], proba[:, cls])
        pr_results[CLASS_NAMES[cls]] = ap
        ax.plot(recall, precision, label='{} (AP={:.4f})'.format(
            CLASS_NAMES[cls], ap))
        print('  {}: Average Precision={:.4f}'.format(CLASS_NAMES[cls], ap))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves - {}'.format(label))
    ax.legend(loc='lower left')
    plt.tight_layout()
    plot_file = 'pr_curves_{}.png'.format(
        label.lower().replace(' ', '_').replace('(', '').replace(')', ''))
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print('  PR curves saved: {}'.format(plot_file))

    with open(results_file, 'a') as f:
        f.write('\n' + '=' * 60 + '\n')
        f.write('CHECK 4: Average Precision Scores ({})\n'.format(label))
        f.write('=' * 60 + '\n')
        for cls_name, ap in pr_results.items():
            f.write('  {}: AP={:.4f}\n'.format(cls_name, ap))

    return pr_results


# ================================================================
# CHECK 5: FULL CLASSIFICATION REPORT
# ================================================================

def check_classification_report(true, pred, results_file, label):
    print('\n' + '=' * 60)
    print('CHECK 5: Classification Report ({})'.format(label))
    print('=' * 60)

    target_names = ['Nothing', 'Cloud'] + [
        'Fog-' + str(i) for i in range(1, MAX_RS - 1)]
    report = classification_report(
        true, pred,
        labels=list(range(MAX_RS)),
        target_names=target_names,
        zero_division=0
    )
    print(report)

    with open(results_file, 'a') as f:
        f.write('\n' + '=' * 60 + '\n')
        f.write('CHECK 5: Classification Report ({})\n'.format(label))
        f.write('=' * 60 + '\n')
        f.write(report + '\n')


# ================================================================
# MAIN
# ================================================================

def main():
    if len(sys.argv) < 2:
        import glob
        files = glob.glob('dataset-*.txt')
        if not files:
            print('[ERROR] No dataset file found.')
            sys.exit(1)
        fname = max(files, key=os.path.getmtime)
        print('[VALIDATION] Auto-found: ' + fname)
    else:
        fname = sys.argv[1]

    if not os.path.exists(fname):
        print('[ERROR] File not found: ' + fname)
        sys.exit(1)

    if not os.path.exists(WEIGHTS_FILE):
        print('[ERROR] Weights not found: ' + WEIGHTS_FILE)
        print('        Run DL_pipeline.py first.')
        sys.exit(1)

    # Results file
    base = os.path.splitext(os.path.basename(fname))[0]
    results_file = 'validation_{}.txt'.format(base)

    with open(results_file, 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('FOGMLS COMPREHENSIVE VALIDATION\n')
        f.write('Dataset: {}\n'.format(fname))
        f.write('Weights: {}\n'.format(WEIGHTS_FILE))
        f.write('=' * 60 + '\n')

    print('\n[VALIDATION] Loading data...')
    X, Y = parse(fname)

    # Train/test split matching DL_pipeline
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_STATE)

    print('[VALIDATION] Loading scaler and model...')
    scaler = joblib.load(SCALER_FILE)
    model, _ = build_model()
    model.load_weights(WEIGHTS_FILE)

    # Get flat per-task features for baseline models
    print('[VALIDATION] Preparing flat features for baseline models...')
    flat_X_train, flat_Y_train = get_active_flat(X_train, Y_train)
    flat_X_test, flat_Y_test = get_active_flat(X_test, Y_test)
    print('[VALIDATION] Active train tasks: {}, test tasks: {}'.format(
        len(flat_X_train), len(flat_X_test)))

    # Get DL predictions
    print('[VALIDATION] Getting DL model predictions...')
    dl_true, dl_pred, dl_proba = get_dl_predictions(model, X_test, Y_test, scaler)
    dl_acc = np.mean(dl_pred == dl_true) * 100
    dl_f1 = f1_score(dl_true, dl_pred, average='macro',
                     labels=ACTIVE_CLASSES, zero_division=0)

    print('\n[VALIDATION] DL Model: accuracy={:.2f}%, macro_f1={:.4f}'.format(
        dl_acc, dl_f1))

    with open(results_file, 'a') as f:
        f.write('\nDL Model: accuracy={:.2f}%, macro_f1={:.4f}\n'.format(
            dl_acc, dl_f1))

    # Run all checks
    baselines = check_baselines(
        flat_X_train, flat_Y_train, flat_X_test, flat_Y_test, results_file)

    cm, errors = check_confusion_matrix(
        dl_true, dl_pred, results_file, 'DL Model')

    roc_results = check_roc(
        dl_true, dl_proba, results_file, 'DL Model')

    pr_results = check_precision_recall(
        dl_true, dl_proba, results_file, 'DL Model')

    check_classification_report(
        dl_true, dl_pred, results_file, 'DL Model')

    # Final summary
    print('\n' + '=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)
    print('DL Model accuracy     : {:.2f}%'.format(dl_acc))
    print('DL Model macro F1     : {:.4f}'.format(dl_f1))
    print()
    print('Baseline comparison:')
    best_baseline_acc = 0
    for name, res in baselines.items():
        gap = dl_acc - res['accuracy']
        print('  {} vs DL: gap={:.2f}pp'.format(name, gap))
        best_baseline_acc = max(best_baseline_acc, res['accuracy'])

    dl_gap = dl_acc - best_baseline_acc
    print()
    if dl_gap > 10:
        print('[STRONG] DL outperforms best baseline by {:.2f}pp'.format(dl_gap))
        print('         DL is learning complex patterns beyond simple rules.')
    elif dl_gap > 3:
        print('[MODERATE] DL outperforms best baseline by {:.2f}pp'.format(dl_gap))
        print('           DL adds value but patterns are partially learnable by simple models.')
    else:
        print('[WEAK] DL only {:.2f}pp above best baseline.'.format(dl_gap))
        print('       Problem may be too simple for DL to add significant value.')

    with open(results_file, 'a') as f:
        f.write('\n' + '=' * 60 + '\n')
        f.write('FINAL SUMMARY\n')
        f.write('=' * 60 + '\n')
        f.write('DL accuracy: {:.2f}%\n'.format(dl_acc))
        f.write('DL macro F1: {:.4f}\n'.format(dl_f1))
        f.write('Best baseline accuracy: {:.2f}%\n'.format(best_baseline_acc))
        f.write('DL gap over best baseline: {:.2f}pp\n'.format(dl_gap))

    print('\n[VALIDATION] Results saved to: {}'.format(results_file))
    print('[VALIDATION] Plots saved as PNG files in working directory.')


if __name__ == '__main__':
    main()