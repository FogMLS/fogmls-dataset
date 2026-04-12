# FogMLS Dataset

A labeled dataset for machine learning based resource scheduling 
in heterogeneous IoT-fog-cloud computing systems.

## About

FogMLS is a labeled dataset containing pre-execution task features 
and optimizer-derived node assignment labels generated using a 
genetic algorithm scheduler across heterogeneous fog and cloud nodes. 
The dataset supports the development and evaluation of supervised 
machine learning models for IoT task scheduling.

## Dataset Download

Available on Zenodo (open access):  
DOI: 10.5281/zenodo.19382240

## Repository Contents

- `data_loader.py` — loads dataset and prepares train/test split
- `train.py` — trains the deep learning scheduling model
- `evaluate.py` — produces F1 and accuracy results
- `figures.py` — generates all figures
- `requirements.txt` — Python dependencies

## How to Use

1. Download the dataset from Zenodo and place it in the same folder as the scripts
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Train the model:
```bash
python train.py fogmls_dataset_seed42.txt
```
4. Evaluate results:
```bash
python evaluate.py fogmls_dataset_seed42.txt
```
5. Generate figures:
```bash
python figures.py fogmls_dataset_seed42.txt
```

**Run order matters:** train.py must run first.
evaluate.py and figures.py depend on files saved by train.py.

## Requirements

Python 3.10, TensorFlow 2.20.0, Keras 3.11.3, 
NumPy 2.2.6, scikit-learn 1.7.1, matplotlib 3.10.8

## References

Harry John and Manjula Shenoy K, A Labeled Dataset for 
Machine Learning Based Resource Scheduling in Heterogeneous 
IoT-Fog-Cloud Systems, Scientific Data (Nature Portfolio), 
under review.

## License

MIT
