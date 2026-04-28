"""
sanity_check.py
===============
Verifies that high DL accuracy is genuine and not an artifact.

Three checks:
1. Random baseline - what accuracy does a random predictor get?
2. Majority class baseline - what if we always predict the most common class?
3. Permuted label test - if we shuffle labels, accuracy should collapse
4. Confusion matrix analysis - are errors systematic or random?

Usage:
    python3 sanity_check.py dataset-YOURFILE.txt
"""

import sys
import os
import glob
import numpy as np
import joblib
from sklearn.metrics import f1_score, confusion_matrix

from config import MAX_JOBS, MAX_RS, FEATURES_PER_JOB
from model_arch import build_model

WEIGHTS_FILE = 'DL_weights_' + str(MAX_JOBS) + '.weights.h5'
SCALER_FILE  = 'DL_scaler_'  + str(MAX_JOBS) + '.joblib'
RANDOM_STATE = 42


# -- helpers ------------------------------------------------------

def get_class_name(cls):
    if cls == 0: return 'Nothing'
    if cls == 1: return 'Cloud'
    return 'Fog-' + str(cls - 1)


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
    print("[PARSE] Rows: {}  Skipped: {}".format(len(rows_X), skipped))
    return np.array(rows_X, dtype=np.float32), np.array(rows_Y, dtype=np.int32)


def get_active(Y_true, Y_pred):
    """Return only active (non-zero label) true and predicted values."""
    all_true, all_pred = [], []
    for job in range(MAX_JOBS):
        t = Y_true[:, job]
        p = Y_pred[:, job]
        mask = t > 0
        if mask.sum() > 0:
            all_true.extend(t[mask].tolist())
            all_pred.extend(p[mask].tolist())
    return np.array(all_true), np.array(all_pred)


def accuracy(true, pred):
    return np.mean(true == pred)


def macro_f1(true, pred):
    active_classes = [c for c in range(1, MAX_RS)]
    return f1_score(true, pred, labels=active_classes,
                    average='macro', zero_division=0)


# -- load data and model -------------------------------------------

def load_everything(fname):
    X, Y = parse(fname)

    # Same split as DL_pipeline.py
    from sklearn.model_selection import train_test_split
    _, X_test, _, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = joblib.load(SCALER_FILE)
    X_test_scaled = scaler.transform(X_test)

    model, _ = build_model()
    model.load_weights(WEIGHTS_FILE)

    return X_test_scaled, Y_test, model


def get_model_predictions(model, X_test):
    preds = model.predict(X_test, verbose=0)
    Y_pred = np.zeros((len(X_test), MAX_JOBS), dtype=np.int32)
    for job in range(MAX_JOBS):
        Y_pred[:, job] = np.argmax(preds[job], axis=1)
    return Y_pred


# -- check 1: random baseline --------------------------------------

def check_random_baseline(Y_test):
    print("\n" + "="*55)
    print("CHECK 1: Random baseline (predicts random valid class)")
    print("="*55)
    rng = np.random.RandomState(42)
    # Random predictor picks uniformly from active classes 1..MAX_RS-1
    Y_rand = rng.randint(1, MAX_RS, size=Y_test.shape)
    true, pred = get_active(Y_test, Y_rand)
    acc  = accuracy(true, pred)
    mf1  = macro_f1(true, pred)
    print("  Random accuracy : {:.2f}%".format(acc * 100))
    print("  Random macro F1 : {:.4f}".format(mf1))
    print("  (Your model should be FAR above this)")
    return acc


# -- check 2: majority class baseline -----------------------------

def check_majority_baseline(Y_test):
    print("\n" + "="*55)
    print("CHECK 2: Majority class baseline (always predicts Fog-1)")
    print("="*55)
    active_mask = Y_test > 0
    active_labels = Y_test[active_mask]
    unique, counts = np.unique(active_labels, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    print("  Majority class: {} ({})".format(
        majority_class, get_class_name(majority_class)))

    Y_maj = np.full(Y_test.shape, majority_class, dtype=np.int32)
    true, pred = get_active(Y_test, Y_maj)
    acc  = accuracy(true, pred)
    mf1  = macro_f1(true, pred)
    print("  Majority accuracy : {:.2f}%".format(acc * 100))
    print("  Majority macro F1 : {:.4f}".format(mf1))
    print("  (Your model should be FAR above this)")
    return acc


# -- check 3: permuted label test ----------------------------------

def check_permuted_labels(model, X_test, Y_test):
    print("\n" + "="*55)
    print("CHECK 3: Permuted label test")
    print("  Shuffles TRUE labels among active positions only.")
    print("  Model accuracy should collapse to ~random level.")
    print("  If it stays high, model is memorizing, not learning.")
    print("="*55)
    rng = np.random.RandomState(42)

    # CRITICAL: permute only active positions.
    # Permuting all positions inflates accuracy because the model
    # correctly predicts Nothing for zero-padded positions (73% of
    # all positions) regardless of label shuffling.
    Y_perm = Y_test.copy()
    for job in range(MAX_JOBS):
        col = Y_perm[:, job]
        active_idx = np.where(col > 0)[0]
        if len(active_idx) > 1:
            shuffled = col[active_idx].copy()
            rng.shuffle(shuffled)
            Y_perm[active_idx, job] = shuffled

    Y_pred = get_model_predictions(model, X_test)

    # Use ORIGINAL Y_test to determine active positions.
    all_true_perm, all_pred = [], []
    for job in range(MAX_JOBS):
        t_orig = Y_test[:, job]
        t_perm = Y_perm[:, job]
        p = Y_pred[:, job]
        mask = t_orig > 0
        if mask.sum() > 0:
            all_true_perm.extend(t_perm[mask].tolist())
            all_pred.extend(p[mask].tolist())
    true_perm = np.array(all_true_perm)
    pred_arr  = np.array(all_pred)

    acc  = accuracy(true_perm, pred_arr)
    mf1  = macro_f1(true_perm, pred_arr)
    print("  Permuted accuracy : {:.2f}%".format(acc * 100))
    print("  Permuted macro F1 : {:.4f}".format(mf1))
    print("  (Should be close to ~25%, NOT close to 99%)")
    return acc


# -- check 4: real model results + confusion matrix ----------------

def check_real_results(model, X_test, Y_test):
    print("\n" + "="*55)
    print("CHECK 4: Real model results + confusion matrix")
    print("="*55)
    Y_pred = get_model_predictions(model, X_test)
    true, pred = get_active(Y_test, Y_pred)

    acc = accuracy(true, pred)
    mf1 = macro_f1(true, pred)
    print("  Model accuracy : {:.2f}%".format(acc * 100))
    print("  Model macro F1 : {:.4f}".format(mf1))

    # Confusion matrix for active classes only
    active_classes = list(range(1, MAX_RS))
    class_names = [get_class_name(c) for c in active_classes]
    cm = confusion_matrix(true, pred, labels=active_classes)

    print("\n  Confusion matrix (rows=true, cols=predicted):")
    header = "  {:>10}".format("") + "".join(
        "{:>10}".format(n) for n in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row = "  {:>10}".format(row_name) + "".join(
            "{:>10}".format(cm[i, j]) for j in range(len(active_classes)))
        print(row)

    print("\n  Confusion matrix interpretation:")
    print("  - Diagonal = correct predictions")
    print("  - Off-diagonal = errors")
    print("  - Common error: Cloud predicted as Fog or vice versa?")

    # Find most common error
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    if cm_no_diag.sum() > 0:
        err_row, err_col = np.unravel_index(
            cm_no_diag.argmax(), cm_no_diag.shape)
        print("  Most common error: true={} predicted as {} ({} times)".format(
            class_names[err_row], class_names[err_col],
            cm_no_diag[err_row, err_col]))
    else:
        print("  No errors found in confusion matrix.")

    return true, pred, cm


# -- summary -------------------------------------------------------

def print_summary(random_acc, majority_acc, permuted_acc, model_acc):
    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    print("  Random baseline   : {:.2f}%".format(random_acc * 100))
    print("  Majority baseline : {:.2f}%".format(majority_acc * 100))
    print("  Permuted labels   : {:.2f}%".format(permuted_acc * 100))
    print("  Your model        : {:.2f}%".format(model_acc * 100))
    print()

    lift = model_acc - majority_acc
    if lift > 0.3:
        print("  [PASS] Model is {:.1f}pp above majority baseline.".format(
            lift * 100))
        print("         Results are genuine.")
    elif lift > 0.1:
        print("  [WARN] Model is only {:.1f}pp above majority baseline.".format(
            lift * 100))
        print("         Results may be driven by class imbalance.")
    else:
        print("  [FAIL] Model barely beats majority baseline.")
        print("         High accuracy is due to class imbalance, not learning.")

    if permuted_acc < random_acc + 0.1:
        print("  [PASS] Permuted label accuracy collapsed as expected.")
        print("         Model is learning real patterns, not memorizing.")
    else:
        print("  [WARN] Permuted accuracy did not collapse enough.")
        print("         Investigate potential data leakage.")


# -- main ----------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        files = glob.glob('dataset-*.txt') + glob.glob('fogmls_dataset_*.txt')
        if not files:
            print("[ERROR] No dataset file found.")
            sys.exit(1)
        fname = max(files, key=os.path.getmtime)
        print("[SANITY] Auto-found: " + fname)
    else:
        fname = sys.argv[1]

    if not os.path.exists(WEIGHTS_FILE):
        print("[ERROR] Weights file not found: " + WEIGHTS_FILE)
        print("        Run DL_pipeline.py first.")
        sys.exit(1)

    print("[SANITY] Loading data and model...")
    X_test, Y_test, model = load_everything(fname)
    print("[SANITY] Test samples: {}".format(len(X_test)))

    random_acc   = check_random_baseline(Y_test)
    majority_acc = check_majority_baseline(Y_test)
    permuted_acc = check_permuted_labels(model, X_test, Y_test)
    true, pred, cm = check_real_results(model, X_test, Y_test)
    model_acc    = accuracy(true, pred)

    print_summary(random_acc, majority_acc, permuted_acc, model_acc)


if __name__ == '__main__':
    main()