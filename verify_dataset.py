"""
verify_dataset.py
=================
Run this after generating a new dataset to confirm it was produced
by the current code and is ready for Random Forest training.

Usage:
    python3 verify_dataset.py fogmls_dataset_3x9_seed42_5000slots.csv

Checks performed:
    1. Row count and column structure (task + slot + label columns)
    2. Label range (must be 0 to MAX_RS-1)
    3. Class distribution in labels
    4. Fraction of active (non-zero) task positions
"""

import sys
import os
import numpy as np

# Constants matching the current dataset schema
MAX_JOBS = 200
MAX_RS = 5
NUM_FOG_NODES = 3
FEATURES_PER_JOB = 6          # task type, instructions, data size,
                               # + live remaining capacity of each of
                               # the 3 fog nodes
SLOT_LEVEL_FEATURES = 11      # per-node CPU, RAM, bandwidth, plus
                               # edge-device and cloud-server latency
SLOT_TIME = 35.0
SENSORS_PER_CLUSTER = 9       # high load; 3 for low, 6 for medium


def parse(fname):
    """Load a deposited dataset CSV (has a header row)."""
    data = np.loadtxt(fname, delimiter=',', skiprows=1)
    task_cols = MAX_JOBS * FEATURES_PER_JOB
    X_task = data[:, :task_cols]
    X_slot = data[:, task_cols:task_cols + SLOT_LEVEL_FEATURES]
    Y = data[:, task_cols + SLOT_LEVEL_FEATURES:].astype(int)
    return X_task, X_slot, Y


def check_schema(X_task, X_slot, Y):
    """Verify the dataset has the expected column structure."""
    expected_task_cols = MAX_JOBS * FEATURES_PER_JOB
    expected_slot_cols = SLOT_LEVEL_FEATURES
    expected_label_cols = MAX_JOBS
    print("\n[CHECK] Dataset schema:")
    print("  Task columns : {} (expected {})".format(
        X_task.shape[1], expected_task_cols))
    print("  Slot columns : {} (expected {})".format(
        X_slot.shape[1], expected_slot_cols))
    print("  Label columns: {} (expected {})".format(
        Y.shape[1], expected_label_cols))

    ok = (X_task.shape[1] == expected_task_cols and
          X_slot.shape[1] == expected_slot_cols and
          Y.shape[1] == expected_label_cols)
    if ok:
        print("  [PASS] Schema matches the current dataset format.")
    else:
        print("  [FAIL] Schema mismatch — this may be an older-format file.")
    return ok


def check_labels(Y):
    print("\n[CHECK] Label range:")
    y_min, y_max = Y.min(), Y.max()
    print("  Min label: {}".format(y_min))
    print("  Max label: {}".format(y_max))
    print("  Valid range: 0 to {}".format(MAX_RS - 1))
    if y_min < 0 or y_max >= MAX_RS:
        print("  [FAIL] Labels out of range.")
        return False
    print("  [PASS] All labels in valid range.")
    return True


def check_class_distribution(Y):
    print("\n[CHECK] Class distribution across all label positions:")
    class_names = ['Padding', 'Cloud', 'Fog-1', 'Fog-2', 'Fog-3']
    unique, counts = np.unique(Y, return_counts=True)
    total = Y.size
    for cls, cnt in zip(unique, counts):
        name = class_names[cls] if cls < len(class_names) else str(cls)
        pct = 100.0 * cnt / total
        print("  Class {:d} ({:8s}): {:8d} ({:.2f}%)".format(
            cls, name, cnt, pct))

    fog_classes = list(range(2, MAX_RS))
    missing = [c for c in fog_classes if c not in unique]
    if missing:
        print("  [WARN] Missing fog classes: {}".format(missing))
    else:
        print("  [PASS] All fog classes present in labels.")

    cloud_count = dict(zip(unique.tolist(), counts.tolist())).get(1, 0)
    cloud_pct = 100.0 * cloud_count / total
    if cloud_pct == 0:
        print("  [NOTE] No Cloud assignments. Expected for low/medium load.")
    elif cloud_pct > 50:
        print("  [WARN] Cloud label fraction is {:.1f}%.".format(cloud_pct))


def check_active_fraction(Y):
    active = (Y > 0).sum()
    total = Y.size
    pct = 100.0 * active / total
    print("\n[CHECK] Active (non-padding) task positions:")
    print("  Active : {:d} / {:d} ({:.2f}%)".format(active, total, pct))


def main():
    if len(sys.argv) < 2:
        print("[ERROR] Pass a dataset filename, e.g.:")
        print("  python3 verify_dataset.py fogmls_dataset_3x9_seed42_5000slots.csv")
        sys.exit(1)
    fname = sys.argv[1]

    if not os.path.exists(fname):
        print("[ERROR] File not found: " + fname)
        sys.exit(1)

    print("[VERIFY] Checking dataset: " + fname)
    print("[VERIFY] File size: {:.1f} MB".format(
        os.path.getsize(fname) / 1e6))

    X_task, X_slot, Y = parse(fname)
    print("\n[CHECK] Parsing:")
    print("  Rows        : {}".format(X_task.shape[0]))
    print("  X_task shape: {}".format(X_task.shape))
    print("  X_slot shape: {}".format(X_slot.shape))
    print("  Y shape     : {}".format(Y.shape))

    ok_schema = check_schema(X_task, X_slot, Y)
    ok_labels = check_labels(Y)
    check_class_distribution(Y)
    check_active_fraction(Y)

    print("\n" + "=" * 50)
    if ok_labels and ok_schema:
        print("DATASET OK.")
    else:
        print("DATASET HAS ISSUES.")
    print("=" * 50)


if __name__ == '__main__':
    main()
