# hadt
Heart Arrhythmia Detection Tools

## Overview
hadt is a Python-based tool that analyzes ECG signals to detect and classify abnormal heartbeats.
It uses various machine learning algorithms to identify various types of arrhythmias using the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).

## Acknowledgements

This project was originally pitched and selected for development as one of the final student projects for Le Wagon's Data Science Bootcamp.
The original version can be visited in the [arrhythmia](https://github.com/fabriciojm/arrhythmia) repository.

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
