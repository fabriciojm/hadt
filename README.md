# hadt
Heart Arrhythmia Detection Tools

## Overview
hadt is a Python-based tool that analyzes ECG signals to detect and classify abnormal heartbeats.
It uses various machine learning algorithms to identify various types of arrhythmias using the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).

Try the web app demo [here](https://fabriciojm-hadt-api.hf.space/).
Besides the web app, the API, models and sample data are all automatically deployed to [HuggingFace](https://huggingface.co/fabriciojm) from this repo.

## Features
- Heartbeat detection and segmentation of heartbeats (to be included). 
- Data preparation and preprocessing.
- Classification of heartbeats in binary (normal vs abnormal) and multiclass cases (e.g. normal, supraventricular, ventricular, other).
- Classification using four approaches:
  - Dynamic Time Warping (DTW).
  - Principal Component Analysis (PCA) and XGBoost.
  - Convolutional Neural Networks (CNN).
  - Long Short-Term Memory (LSTM) networks.


## Summary of results

Classification models were evaluated using the accuracy score, on the multiclass case.
We dropped the 'F' category, that represented less than 1% of the data, and kept the N, S, V and Q categories.
Data was undersampled to achieve class balance.

- LSTM: 96.7%
- CNN: 95.9%
- PCA + XGBoost: 95.7%
- DTW: 94.8%


## Installation (to be added)

Use the standard pip install for the package:

```bash
cd hadt
pip install .
```

## Usage

```bash
python hadt/ml_logic/[binary/multiclass]/[MODEL_NAME].py --config config/[binary/multiclass].json
```
Where `[MODEL_NAME]` is the name of the model to be used, e.g. `lstm_multi_model.h5` and selecting one of the `[binary/multiclass]` options according to the classification task.


## Acknowledgements

- This project was originally pitched and selected for development as one of the final student projects for Le Wagon's Data Science Bootcamp.
The original version can be visited in the [arrhythmia](https://github.com/fabriciojm/arrhythmia) repository.

- Preprocessed data was taken from [this Kaggle public dataset](https://www.kaggle.com/datasets/talal92/mit-bih-dataset-preprocess-into-heartbeat-python), which is derived from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).

- The 1D CNN model was taken from "Classifying Cardiac Arrhythmia from ECG Signal Using 1D CNN Deep Learning Model" by Adel A. Ahmed 1,Waleed Ali,Talal A. A. Abdullah  and Sharaf J. Malebary [link](https://www.mdpi.com/2227-7390/11/3/562).


