"""
zero_shot_transfer_test.py
============================
Loads the ALREADY-TRAINED traffic-surveillance RF model (no retraining)
and tests it directly on the healthcare/patient-monitoring dataset.

Usage:
    python3 zero_shot_transfer_test.py <traffic_model.joblib> <healthcare_dataset.csv>
"""
import sys
import numpy as np
import joblib
from sklearn.metrics import classification_report, f1_score

def load_and_reshape(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    X_task = data[:, :1200].reshape(-1, 200, 6)
    Y = data[:, 1211:].astype(int)

    flat_X, flat_Y = [], []
    for row_idx in range(X_task.shape[0]):
        active_mask = Y[row_idx] != 0
        active_tasks = X_task[row_idx][active_mask]
        active_labels = Y[row_idx][active_mask]
        queue_positions = np.arange(len(active_tasks))
        rows = np.column_stack([active_tasks, queue_positions])
        flat_X.append(rows)
        flat_Y.append(active_labels)
    flat_X = np.vstack(flat_X)
    flat_Y = np.concatenate(flat_Y)
    return flat_X, flat_Y


def main():
    model_path = sys.argv[1]
    healthcare_csv = sys.argv[2]

    print(f"[ZERO-SHOT] Loading already-trained traffic model: {model_path}")
    model = joblib.load(model_path)

    print(f"[ZERO-SHOT] Loading healthcare dataset: {healthcare_csv}")
    X_health, Y_health = load_and_reshape(healthcare_csv)
    print(f"[ZERO-SHOT] Healthcare task-level rows: {len(X_health)}")

    print("[ZERO-SHOT] Predicting on healthcare data using the UNMODIFIED traffic model (no retraining)...")
    Y_pred = model.predict(X_health)

    acc = np.mean(Y_pred == Y_health) * 100
    macro_f1 = f1_score(Y_health, Y_pred, average='macro', zero_division=0)

    print(f"\n[ZERO-SHOT] Zero-shot accuracy (traffic model on healthcare data): {acc:.2f}%")
    print(f"[ZERO-SHOT] Macro F1: {macro_f1:.4f}")
    print()
    print(classification_report(Y_health, Y_pred, zero_division=0,
                                 target_names=['Padding','Cloud','Fog-1','Fog-2','Fog-3'],
                                 labels=[0,1,2,3,4]))

if __name__ == '__main__':
    main()
