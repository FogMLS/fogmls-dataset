"""
Example conversion utilities for the FogMLS dataset:
1. Tabular -> multi-instance ("bag of tasks") representation
2. Tabular -> per-slot bipartite graph representation

These convert the fixed-width tabular format into two commonly requested
alternative representations for machine learning research.
"""
import numpy as np
import networkx as nx


def load_tabular(path, n_task_positions=200, n_task_features=6, n_slot_features=11):
    """Load the raw dataset and split into task-level, slot-level, and label arrays."""
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    n_task_cols = n_task_positions * n_task_features
    X_task = data[:, :n_task_cols].reshape(-1, n_task_positions, n_task_features)
    X_slot = data[:, n_task_cols:n_task_cols + n_slot_features]
    Y = data[:, n_task_cols + n_slot_features:].astype(int)
    return X_task, X_slot, Y


def to_multi_instance(X_task, Y):
    """
    Convert fixed-width tabular rows into a multi-instance ("bag of tasks")
    representation: one variable-length bag of active task feature vectors
    per slot, with its corresponding label sequence, padding removed.

    Returns a list of (bag_features, bag_labels) tuples, one per slot.
    """
    bags = []
    n_slots = X_task.shape[0]
    for slot_idx in range(n_slots):
        active_mask = Y[slot_idx] != 0
        bag_features = X_task[slot_idx][active_mask]
        bag_labels = Y[slot_idx][active_mask]
        bags.append((bag_features, bag_labels))
    return bags


RESOURCE_NAMES = {1: 'Cloud', 2: 'Fog-1', 3: 'Fog-2', 4: 'Fog-3'}


def to_slot_graph(X_task, X_slot, Y, slot_idx):
    """
    Build a bipartite graph for one scheduling slot: task nodes on one side,
    resource nodes (Cloud, Fog-1, Fog-2, Fog-3) on the other, with an edge
    connecting each active task to the resource it was assigned to.

    Returns a networkx.Graph with task feature vectors and slot-level
    context stored as node/graph attributes.
    """
    G = nx.Graph()
    for r_id, r_name in RESOURCE_NAMES.items():
        G.add_node(r_name, kind='resource')

    active_mask = Y[slot_idx] != 0
    active_task_indices = np.where(active_mask)[0]

    for task_pos in active_task_indices:
        task_node = f"task_{task_pos}"
        features = X_task[slot_idx, task_pos]
        label = Y[slot_idx, task_pos]
        G.add_node(task_node, kind='task', features=features.tolist())
        G.add_edge(task_node, RESOURCE_NAMES[label])

    G.graph['slot_features'] = X_slot[slot_idx].tolist()
    return G


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'dataset-13-09-202617-17-58.txt'

    X_task, X_slot, Y = load_tabular(path)
    print(f"Loaded: {X_task.shape[0]} slots, {X_task.shape[1]} task positions, "
          f"{X_task.shape[2]} task features, {X_slot.shape[1]} slot features")

    bags = to_multi_instance(X_task, Y)
    print(f"\nMulti-instance conversion: {len(bags)} bags")
    print(f"  Slot 0: {bags[0][0].shape[0]} active tasks, "
          f"feature shape {bags[0][0].shape}, label shape {bags[0][1].shape}")

    G = to_slot_graph(X_task, X_slot, Y, slot_idx=0)
    print(f"\nGraph conversion for slot 0: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    print(f"  Node kinds: {set(nx.get_node_attributes(G, 'kind').values())}")
    print(f"  Sample resource node degree: Cloud has degree {G.degree('Cloud')}")
