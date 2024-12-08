# arrhythmia
A tool for identifying anomalous heartbeats
## Overview
Arrhythmia is a Python-based tool that analyzes ECG signals to detect and classify abnormal heartbeats.
It uses machine learning algorithms to identify various types of arrhythmias using the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).

A web demo can be found in here (add link).

It started as a project for [Le Wagon](https://www.lewagon.com/)'s Data Science Bootcamp.
Developed by [Chloé Avenas](https://github.com/Chlouette), [Lucas France](https://github.com/Lucasfilm360) and [Fabricio Jiménez Morales](https://github.com/fabriciojm).

## Features
- Heartbeat detection and segmentation of heartbeats (to be included). 
- Data preparation and preprocessing.
- Classification of heartbeats in binary (normal vs abnormal) and multiclass cases (e.g. normal, supraventricular, ventricular, other).
- Classification using four approaches:
  - Dynamic Time Warping (DTW).
  - Principal Component Analysis (PCA) and XGBoost.
  - Convolutional Neural Networks (CNN).
  - Long Short-Term Memory (LSTM) networks.


## Summary of results (to be included)



## Installation
To install Arrhythmia, follow these steps:

1. Ensure you have Python 3.8 or higher installed
1. Clone the repository:
```bash
git clone https://github.com/fabriciojm/arrhythmia.git
cd arrhythmia
```

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
1. Install required dependencies:
```bash
pip install -r requirements.txt
```
1. Verify installation:
```bash
python -m arrhythmia --version
```
