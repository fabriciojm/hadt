import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Can do smarter imports
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import os, argparse, json, time
from hadt.ml_logic.preproc import preproc, label_encoding


def initialize_model():
    return Sequential([
        Conv1D(8, 3, activation='relu', input_shape=(180, 1)),
        Conv1D(16, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),
        Flatten(),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

def compile_model(model):
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy','recall','precision'])
    return model

def train_model(model, df_tr):
    X_tr, y_tr = df_tr.drop(columns='target'), df_tr.target
    X_tr = np.expand_dims(X_tr, axis = -1)

    es = EarlyStopping(patience = 5, restore_best_weights=True)
    history = model.fit(X_tr, y_tr,
                        epochs=50, batch_size=32,
                        validation_split=0.2,
                        callbacks=[es])
    return model, history

def save_model(model, path):
    model.save(path)

def evaluate_model(model, df_te):
    X_te, y_te = df_te.drop(columns='target'), df_te.target
    X_te = np.expand_dims(X_te, axis=-1)  # Note: Different axis for CNN
    y_pred = model.predict(X_te)
    y_pred = np.round(y_pred)
    print(classification_report(y_te, y_pred, digits=4))
    print(confusion_matrix(y_te, y_pred))
    return model.evaluate(X_te, y_te)

if __name__=="__main__":
# Wagon production code
#     if MODEL_TARGET == "gcs":
#             model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
#             client = storage.Client()
#             bucket = client.bucket(BUCKET_NAME)
#             blob = bucket.blob(f"models/{model_filename}")
#             blob.upload_from_filename(model_path)

#             print("✅ Model saved to GCS")
#             return None

#     if MODEL_TARGET == "mlflow":
#         mlflow.tensorflow.log_model(
#             model=model,
#             artifact_path="model",
#             registered_model_name=MLFLOW_MODEL_NAME
#         print("✅ Model saved to MLflow")
#         return None
#     return None

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    parser = argparse.ArgumentParser(description='Train CNN model for binary classification')
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

    # Load config file first
    config = defaults.copy()
    if args.config:
        with open(args.config, 'r') as file:
            config.update(json.load(file))

    # Override config with CLI arguments
    for key, value in vars(args).items():
        if (key != 'config' and 
            value is not None and 
            not (isinstance(value, bool) and value == False)):
            config[key] = value

    print(f"Using final config:\n{config}")

    # Prepare file paths
    label_encoding_path = os.path.join(config['output_dir'], 'cnn_binary_label_encoding.pkl')
    model_path = os.path.join(config['output_dir'], 'cnn_binary_model.h5')

    # Load and preprocess data
    df = pd.read_csv(config['filename'])
    
    # Extract preprocessing parameters from config
    preproc_kwargs = {
        'drop_classes': config['drop_classes'],
        'n_samples': config['n_samples'],
        'binary': config['binary'],
        'scaler_name': config['scaler_name']
    }
    
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
