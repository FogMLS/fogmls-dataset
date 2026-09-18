# FogMLS Dataset — Validation Scripts

This repository contains the validation and analysis scripts supporting
the FogMLS dataset paper, *"A labeled dataset for resource scheduling in
heterogeneous IoT-fog-cloud computing systems"* (submitted to *Scientific
Data*).

The deposited datasets themselves are hosted separately on Zenodo:
**https://doi.org/10.5281/zenodo.22762150**

## What this repository contains

This repository provides the scripts used to validate the deposited
datasets and reproduce every result reported in the paper. It does
**not** contain the FogMLS simulator or GA-based dataset-generation code.
That code is described in a companion manuscript currently under review
and will be released publicly upon that manuscript's acceptance (see the
paper's Code Availability section). Until then, it is available to
editors and reviewers on request.

## Requirements

- Python 3.10.21
- scikit-learn 1.7.1
- NumPy 2.2.6
- joblib 1.6.0
- networkx (for `convert_formats.py` only)

Install with:
```
pip install scikit-learn==1.7.1 numpy==2.2.6 joblib==1.6.0 networkx
```

No GPU is required; all scripts run on CPU.

## Getting the data

Download the three deposited datasets from Zenodo
(https://doi.org/10.5281/zenodo.22762150) before running any script here:
- `fogmls_dataset_3x3_seed42_5000slots.csv` (low load)
- `fogmls_dataset_3x6_seed42_5000slots.csv` (medium load)
- `fogmls_dataset_3x9_seed42_5000slots.csv` (high load, primary dataset)

Place them in this repository's root directory, or pass the file path
directly as an argument to each script below.

## Scripts

### `run_rf_named.py`
Trains and evaluates the primary Random Forest classifier on a given
dataset. Reports overall accuracy and a full classification report
(precision, recall, F1 per class). This produces the primary accuracy
figures reported throughout the paper.
```
python3 run_rf_named.py <dataset.csv> <model_output.joblib> <results_log.csv>
```

### `decision_tree_baseline.py`
Trains Decision Tree classifiers (depth 3 and depth 10) on the same
task-level features as the Random Forest model, for the baseline
comparison in Table (Baseline Comparison).

### `cross_seed_eval.py`
Evaluates a model trained on one random seed against data generated
under a different seed, to check cross-seed generalization.

### `permutation_test.py`
Shuffles true labels and re-evaluates classification accuracy, to check
for data leakage. A properly leakage-free model should collapse to
near-chance accuracy under permutation.

### `per_class_metrics.py`
Computes per-class precision, recall, F1, and ROC AUC for the trained
Random Forest model.

### `actuation_split_check.py`
Compares classification accuracy on actuation tasks (zero instruction
count) against regular tasks, to check that Label 0 padding and
actuation tasks are not being conflated in evaluation.

### `convert_formats.py`
Example utility functions converting the fixed-width tabular dataset
format into two alternative representations that may be more convenient
for certain machine learning approaches:
- **Multi-instance ("bag of tasks")**: a variable-length bag of active
  task feature vectors per slot, with padding removed.
- **Per-slot bipartite graph**: a `networkx` graph with task nodes
  connected to the resource node (Cloud, Fog-1, Fog-2, Fog-3) they were
  assigned to, useful for graph-based ML approaches (e.g. GNNs).

```
python3 convert_formats.py fogmls_dataset_3x9_seed42_5000slots.csv
```

## Reproducing the paper's results

Each table and figure in the paper can be reproduced by running the
corresponding script above against the relevant deposited dataset. See
the Methods and Technical Validation sections of the paper for the exact
configuration (random seed, train/test split) used for each reported
result.

## Citation

If you use this code or the associated datasets, please cite the paper
(citation details to be added upon publication) and the Zenodo dataset
deposit (https://doi.org/10.5281/zenodo.22762150).

## License

Released under the Creative Commons CC BY 4.0 license, matching the
license of the deposited datasets.

## Contact

For questions, or to request early access to the simulator and
GA-based dataset-generation code prior to its public release, please
contact the corresponding author.
