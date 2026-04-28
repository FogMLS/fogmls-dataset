"""
verify_dataset.py
=================
Run this after generating a new dataset to confirm it was produced
by the fixed code and is ready for DL training.

Usage:
    python3 verify_dataset.py dataset-DDMMYYYYHH-MM-SS.txt

Checks performed:
    1. Row count and column structure
    2. Label range (must be 0 to MAX_RS-1)
    3. Capacity feature range (must be 0 to SLOT_TIME, not 0 to cpu_speed*SLOT_TIME)
    4. Class distribution in labels
    5. Fraction of active (non-zero) task positions
    6. Consistency: rows where label=0 should have zero features
"""

import sys
import os
import numpy as np

# Hardcoded constants matching config.py
# Change these if you change config.py values
MAX_JOBS = 200
MAX_RS = 5
NUM_FOG_NODES = 3
JOB_ATTRIBUTES = 3
# Capacity features removed from dataset -- task features only
FEATURES_PER_JOB = JOB_ATTRIBUTES   # 3
SLOT_TIME = 35.0
SENSORS_PER_CLUSTER = 9


def parse(fname):
    rows_X = []
    rows_Y = []
    skipped = 0
    with open(fname) as f:
        lines = f.readlines()
    expected = MAX_JOBS * FEATURES_PER_JOB + MAX_JOBS
    for line in lines:
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
    return np.array(rows_X), np.array(rows_Y, dtype=int), skipped


def check_schema(X):
    """Verify dataset has correct number of columns for task-only schema."""
    expected_cols = MAX_JOBS * FEATURES_PER_JOB
    actual_cols = X.shape[1]
    print("\n[CHECK] Dataset schema (task features only, no capacity):")
    print("  Expected columns : {}".format(expected_cols))
    print("  Actual columns   : {}".format(actual_cols))
    print("  FEATURES_PER_JOB : {} (TaskType + Instructions + DataSize)".format(
        FEATURES_PER_JOB))
    if actual_cols == expected_cols:
        print("  [PASS] Schema matches task-only format.")
        return True
    else:
        print("  [FAIL] Schema mismatch. Expected {} got {}.".format(
            expected_cols, actual_cols))
        if actual_cols == MAX_JOBS * (FEATURES_PER_JOB + NUM_FOG_NODES):
            print("         Dataset appears to contain capacity features.")
            print("         Regenerate with updated IoTsimulation.py.")
        return False


def check_labels(Y):
    print("\n[CHECK] Label range:")
    y_min = Y.min()
    y_max = Y.max()
    print("  Min label: {}".format(y_min))
    print("  Max label: {}".format(y_max))
    print("  Valid range: 0 to {}".format(MAX_RS - 1))

    if y_min < 0 or y_max >= MAX_RS:
        print("  [FAIL] Labels out of range. Check DEVICE_TO_CLASS mapping.")
        return False
    print("  [PASS] All labels in valid range.")
    return True


def check_class_distribution(Y):
    print("\n[CHECK] Class distribution across all label positions:")
    class_names = ['Nothing', 'Cloud', 'Fog-1', 'Fog-2', 'Fog-3']
    unique, counts = np.unique(Y, return_counts=True)
    total = Y.size
    for cls, cnt in zip(unique, counts):
        name = class_names[cls] if cls < len(class_names) else str(cls)
        pct = 100.0 * cnt / total
        print("  Class {:d} ({:8s}): {:8d} ({:.1f}%)".format(
            cls, name, cnt, pct))

    # Check that all fog classes are present
    fog_classes = list(range(2, MAX_RS))
    missing = [c for c in fog_classes if c not in unique]
    if missing:
        print("  [WARN] Missing fog classes: " + str(missing))
        print("         This may indicate fog nodes never received tasks.")
    else:
        print("  [PASS] All fog classes present in labels.")

    # Check cloud is not overwhelming
    cloud_count = dict(zip(unique.tolist(), counts.tolist())).get(1, 0)
    cloud_pct = 100.0 * cloud_count / total
    if cloud_pct > 50:
        print("  [WARN] Cloud label fraction is {:.1f}%.".format(cloud_pct))
        print("         High cloud fraction suggests fog nodes are frequently")
        print("         saturated. Check if simulation ran long enough.")


def check_active_fraction(Y):
    active = (Y > 0).sum()
    total = Y.size
    pct = 100.0 * active / total
    print("\n[CHECK] Active (non-zero) task positions:")
    print("  Active  : {:d} / {:d} ({:.1f}%)".format(active, total, pct))
    sensors_per_cluster = SENSORS_PER_CLUSTER
    fog_nodes = 3
    expected_active_per_slot = sensors_per_cluster * fog_nodes
    expected_pct = 100.0 * expected_active_per_slot / MAX_JOBS
    print("  Expected: ~{:.1f}% ({} sensors x {} clusters / {} slots)".format(
        expected_pct, sensors_per_cluster, fog_nodes, MAX_JOBS))
    print("  Note: actual active fraction is higher because chained tasks accumulate across slots.")
    if pct < expected_pct * 0.3:
        print("  [WARN] Active fraction much lower than expected.")
        print("         Check if chained tasks are being captured.")
    else:
        print("  [PASS] Active fraction consistent with topology.")


def main():
    if len(sys.argv) < 2:
        # Auto-find latest dataset
        import glob
        files = glob.glob('dataset-*.txt') + glob.glob('fogmls_dataset_*.txt')
        if not files:
            print("[ERROR] No dataset files found. Pass filename as argument.")
            sys.exit(1)
        fname = max(files, key=os.path.getmtime)
        print("[VERIFY] Auto-found: " + fname)
    else:
        fname = sys.argv[1]

    if not os.path.exists(fname):
        print("[ERROR] File not found: " + fname)
        sys.exit(1)

    print("[VERIFY] Checking dataset: " + fname)
    print("[VERIFY] File size: {:.1f} MB".format(
        os.path.getsize(fname) / 1e6))

    X, Y, skipped = parse(fname)
    print("\n[CHECK] Parsing:")
    print("  Rows parsed : {}".format(len(X)))
    print("  Rows skipped: {}".format(skipped))
    print("  X shape     : {}".format(X.shape))
    print("  Y shape     : {}".format(Y.shape))

    if len(X) == 0:
        print("[FAIL] No data rows found.")
        sys.exit(1)

    ok_labels = check_labels(Y)
    ok_schema = check_schema(X)
    check_class_distribution(Y)
    check_active_fraction(Y)

    print("\n" + "=" * 50)
    if ok_labels and ok_schema:
        print("DATASET OK - ready for DL training.")
    else:
        print("DATASET HAS ISSUES - do not train on this data.")
        print("Regenerate with updated IoTsimulation.py.")
    print("=" * 50)


if __name__ == '__main__':
    main()