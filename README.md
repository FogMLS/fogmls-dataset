# FogMLS Dataset

A labeled dataset for resource scheduling in heterogeneous 
IoT-fog-cloud computing systems.

## About

FogMLS is a labeled dataset containing pre-execution task features 
and optimizer-derived node assignment labels generated using a genetic 
algorithm scheduler across three heterogeneous fog nodes and a cloud 
server. The dataset covers three load scenarios (low, medium, high) 
and supports supervised machine learning, reinforcement learning 
initialisation, and reproducible benchmarking for IoT task scheduling.

## Dataset Download

Available on Zenodo (open access):  
DOI: PENDING

## Repository Contents

- `DL_pipeline.py` — trains the deep learning validation model
- `comprehensive_validation.py` — baseline comparison and ROC/PR curves
- `sanity_check.py` — permutation test and majority baseline checks
- `verify_dataset.py` — structural integrity checks
- `diagnose_features.py` — feature separability analysis
- `model_arch.py` — shared model architecture definition
- `config.py` — simulation configuration (3x9 high load scenario)

## How to Use

1. Download the dataset from Zenodo and place it in the same folder
2. Install dependencies:

```bash
pip install tensorflow==2.20.0 keras==3.11.3 numpy==2.2.6 scikit-learn==1.7.1 matplotlib
```

3. Train the model:

```bash
python DL_pipeline.py fogmls_dataset_3x9_seed42_5000slots.csv
```

4. Run validation:

```bash
python comprehensive_validation.py fogmls_dataset_3x9_seed42_5000slots.csv
python sanity_check.py fogmls_dataset_3x9_seed42_5000slots.csv
```

**Run order matters:** `DL_pipeline.py` must run first. 
`comprehensive_validation.py` and `sanity_check.py` depend on 
the weights file saved by `DL_pipeline.py`.

## Requirements

Python 3.10, TensorFlow 2.20.0, Keras 3.11.3,  
NumPy 2.2.6, scikit-learn 1.7.1, matplotlib 3.10.8

## References

Harry John and Manjula Shenoy K, A Labeled Dataset for Resource 
Scheduling in Heterogeneous IoT-Fog-Cloud Computing Systems, 
Scientific Data (Nature Portfolio), under review.

## License

MIT
