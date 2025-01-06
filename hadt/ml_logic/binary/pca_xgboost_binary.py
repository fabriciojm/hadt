import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time
import joblib, os, argparse
from sklearn.utils.validation import check_is_fitted

from hadt.ml_logic.preproc import preproc, label_encoding
from hadt.ml_logic.utils.config_utils import build_config


def initialize_model(k):
    return make_pipeline(PCA(n_components=k), XGBClassifier())

def compile_model(model):
    # Comply with logic of other models
    return model

def train_model(model, df_tr):
    X_tr, y_tr = df_tr.drop(columns='target'), df_tr.target
    model.fit(X_tr, y_tr)
    return model, None

def evaluate_model(model, df_te):
    X_te, y_te = df_te.drop(columns='target'), df_te.target
    
    y_pred = model.predict(X_te)
    print(classification_report(y_te, y_pred, digits=4))
    print(confusion_matrix(y_te, y_pred))
    return model.score(X_te, y_te)

def save_model(model, model_path):
    joblib.dump(model, model_path)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    parser = argparse.ArgumentParser(description='Train PCA-XGBoost model for binary classification')
    parser.add_argument('--config', type=str, help='Path to the model configuration file')
    parser.add_argument('--filename', type=str, help='Path to the input CSV file')
    parser.add_argument('--drop_classes', nargs='+', choices=['N', 'F', 'Q', 'S', 'V'], 
                       help='List of classes to drop (N, F, Q, S, V)')
    parser.add_argument('--output_dir', type=str, help='Directory to save model and encodings')
    parser.add_argument('--n_samples', type=int, help='Number of samples to use for training')
    parser.add_argument('--binary', type=bool, help='Whether to use binary classification')
    parser.add_argument('--scaler_name', type=str, help='Name of the scaler to use')
    parser.add_argument('--k', type=int, help='Number of principal components to use')


    args = parser.parse_args()

    defaults = {
        'filename': '../arrhythmia_mit_bih/MIT-BIH.csv',
        'drop_classes': ['F'],
        'output_dir': os.getcwd(),
        'n_samples': -1,
        'binary': True,
        'scaler_name': 'MeanVariance',
        'k': 8
    }

    config, preproc_kwargs = build_config(args, defaults=defaults)


    # Prepare file paths
    label_encoding_path = os.path.join(config['output_dir'], 'pca_xgboost_binary_label_encoding.pkl')
    model_path = os.path.join(config['output_dir'], 'pca_xgboost_binary_model.pkl')

    df = pd.read_csv(config['filename'])
    df_tr, df_te = preproc(df, **preproc_kwargs)
    df_tr, df_te = label_encoding([df_tr, df_te], label_encoding_path)

    model = initialize_model(config['k'])
    model = compile_model(model)
    model, _ = train_model(model, df_tr)

    save_model(model, model_path)
    evaluate_model(model, df_te)