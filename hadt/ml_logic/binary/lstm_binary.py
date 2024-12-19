import numpy as np
import pandas as pd
import os
import time
import argparse
import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

from hadt.ml_logic.preproc import preproc, df_from_bucket, label_encoding
from hadt.ml_logic.utils.config_utils import build_config

def initialize_model():
    return Sequential([Input((1, 180)),
                       LSTM(64),
                       Dense(1, activation='sigmoid')])

def compile_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, df_tr):
    X_tr, y_tr = df_tr.drop(columns='target'), df_tr.target
    X_tr = np.expand_dims(X_tr, axis = 1)

    es = EarlyStopping(patience = 10, restore_best_weights=True)
    history = model.fit(X_tr, y_tr,
                        epochs=100, batch_size=256,
                        validation_split=0.2,
                        callbacks=[es])

    return model, history

def save_model(model, path):
    model.save(path)

def evaluate_model(model, df_te):
    X_te, y_te = df_te.drop(columns='target'), df_te.target
    X_te = np.expand_dims(X_te, axis = 1)
    y_pred = model.predict(X_te)
    y_pred = np.round(y_pred)
    print(classification_report(y_te, y_pred, digits=4))
    print(confusion_matrix(y_te, y_pred))
    return model.evaluate(X_te, y_te)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    parser = argparse.ArgumentParser(description='Train LSTM model for binary classification')
    parser.add_argument('--config', type=str, help='Path to the model configuration file')
    parser.add_argument('--filename', type=str, help='Path to the input CSV file')
    parser.add_argument('--drop_classes', nargs='+', choices=['N', 'F', 'Q', 'S', 'V'], 
                       help='List of classes to drop (N, F, Q, S, V)')
    parser.add_argument('--output_dir', type=str, help='Directory to save model and encodings')
    parser.add_argument('--n_samples', type=int, help='Number of samples to use for training')
    parser.add_argument('--binary', type=bool, help='Whether to use binary classification')
    parser.add_argument('--scaler_name', type=str, help='Name of the scaler to use')
    
    args = parser.parse_args()

    # Define default values
    defaults = {
        'filename': '../arrhythmia_raw_data/MIT-BIH_raw.csv',
        'drop_classes': ['F'],
        'output_dir': os.getcwd(),
        'n_samples': -1,
        'binary': True, 
        'scaler_name': 'MeanVariance'
    }
    
    config, preproc_kwargs = build_config(args, defaults=defaults)

    # Prepare file paths
    label_encoding_path = os.path.join(config['output_dir'], 'lstm_binary_label_encoding.pkl')
    model_path = os.path.join(config['output_dir'], 'lstm_binary_model.h5')

    # Load and preprocess data
    df = pd.read_csv(config['filename'])
    df_tr, df_te = preproc(df, **preproc_kwargs)
    df_tr, df_te = label_encoding([df_tr, df_te], label_encoding_path)

    # Train and evaluate model
    t_start = time.time()
    model = initialize_model()
    model = compile_model(model)
    model, history = train_model(model, df_tr)
    t_end = time.time()
    print(f"It took {t_end - t_start} seconds for the model to make this prediction.")

    save_model(model, model_path)
    evaluate_model(model, df_te)
