import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tslearn.utils import to_time_series_dataset
from imblearn.over_sampling import SMOTE

import argparse, os, io, json, pickle
from google.cloud import storage

def df_from_bucket(key_path, bucket_name='arrhythmia_mit_bih', file_name='MIT-BIH.csv'):

    print(f'Getting {file_name} from {bucket_name} gcp bucket.')
    # Set the environment variable for the service account key
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download as bytes
    content = blob.download_as_bytes()

    # Convert to pandas DataFrame (for CSV files)
    df = pd.read_csv(io.BytesIO(content))
    return df

def label_encoding(dfs, path):
    # dfs is a list of dataframes because it could be train/test or train/val/test
    le = LabelEncoder()
    le.fit(pd.concat(dfs, axis=0).target)
    for df in dfs:
        df['target'] = le.transform(df.target)
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    with open(path, "wb") as f:
        pickle.dump(mapping, f)
    print('Encoding:', mapping)
    print(f"Encoding saved to '{path}'")
    return dfs

def label_decoding(value, path):
    """
    path is a file path for the encoding dictionary
    """
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    inverse_mapping = {v: k for k, v in mapping.items()}
    return inverse_mapping[value]

def undersample(X, y, n_samples):
    # if n_samples is -1, undersampling to size of least common class
    if n_samples == -1:
        # Get counts of each unique value and find minimum
        _, counts = np.unique(y, return_counts=True)
        n_samples = np.min(counts)
        # n_samples = y.value_counts().sort_values(ascending=False).iloc[-1]
    # Get indices for each class
    class_indices = {}
    for class_label in np.unique(y):
        class_mask = (y == class_label)
        indices = np.where(class_mask)[0]
        # Randomly select n_samples indices for this class
        selected = np.random.choice(indices, size=n_samples, replace=False)
        class_indices[class_label] = selected

    # Combine all selected indices and sort them
    selected_indices = np.sort(np.concatenate(list(class_indices.values())))

    # Select the samples using the indices
    X = X[selected_indices]
    y = y[selected_indices]

    return X, y

def pandasify(X, y, ts_features):
    X = pd.DataFrame(X.squeeze(), columns=ts_features)
    y = pd.Series(y, name='target')
    return pd.concat([X, y], axis=1)



def preproc_xgb_single(filepath, pca_model_path="/home/fabricio/pca_multiclass.pkl", scaler_name='MeanVariance'):
    X = pd.read_csv(filepath)
    if X.shape != (1, 180):
        print('File shape is not (1, 180) but ', X.shape, '. Exiting')
        return

    X = to_time_series_dataset(X)
    X = X.reshape(X.shape[0], -1)
    if scaler_name == "MeanVariance":
        scaler = TimeSeriesScalerMeanVariance()
    elif scaler_name == "MinMax":
        scaler = TimeSeriesScalerMinMax()
    else:
        print(f"Error: {scaler_name} not known.")
        return
    X = scaler.fit_transform(X)

    with open(pca_model_path, "rb") as file:
        pca = pickle.load(file)
    shape = X.shape
    X = pca.transform(X.reshape(1, 180))
    return X

def preproc(df, n_samples=-1, drop_classes=[], binary=False, scaler_name='MeanVariance'):

    # Create a copy to avoid funny errors
    df = df.copy()

    if drop_classes == []:
        print('Preprocess keeping all classes')
    else:
        print(f'Preprocess dropping {drop_classes} class(es)')
    if binary:
        print("Preparing binary.")

    # Dropping redundant index
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Original encoding of classes in MIT-BIH .csv
    enc = {'N': 1, 'F': 0, 'Q': 2, 'S': 3, 'V': 4}
    inv_enc = {v: k for k, v in enc.items()}
    # Drop rows where target is in drop_classes (using loc for proper assignment)
    enc_drop_classes = [enc[c] for c in drop_classes]
    df = df.loc[~df['target'].isin(enc_drop_classes)].reset_index(drop=True)
    df['target'] = df['target'].apply(lambda x: inv_enc[x])

    # group data into two classes if binary
    if binary:
        df['target'] = df['target'].apply(lambda x: 'A' if x != 'N' else 'N')

    # Reshape data for tslearn (samples, timestamps, features)
    X = to_time_series_dataset(df.drop(columns=['target']))
    y = df['target'].values

    # Reshape
    X = X.reshape(X.shape[0], -1)

    # Balance classes
    X, y = undersample(X, y, n_samples)

    # Scale the time series, split
    if scaler_name == "MeanVariance":
        scaler = TimeSeriesScalerMeanVariance()
    elif scaler_name == "MinMax":
        scaler = TimeSeriesScalerMinMax()
    else:
        print(f"Error: {scaler_name} not known.")
        return

    # train_test_split, scale
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    ts_features = df.drop(columns='target').columns
    df_tr = pandasify(X_tr, y_tr, ts_features)
    df_te = pandasify(X_te, y_te, ts_features)

    print('Train dims ', X_tr.shape, y_tr.shape)
    print('Test dims ', X_te.shape, y_te.shape)
    return df_tr, df_te

def prepare_filename(**kwargs):
    # Extract required arguments from kwargs
    filename = kwargs.get("filename", "default.csv")
    binary = kwargs.get("binary", False)
    scaler_name = kwargs.get("scaler_name", "MeanVariance")
    drop_classes = kwargs.get("drop_classes", [])
    output_dir = kwargs.get("output_dir", "./")

    # Prepare the filename
    out_filename = os.path.basename(filename)[:-4]
    out_filename += '_preproc'

    if binary:
        out_filename += '_binary'
    out_filename += f'_{scaler_name}'
    if drop_classes:
        out_filename += f"_drop{''.join(drop_classes)}"
    out_filename = os.path.join(output_dir, out_filename)

    return out_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the data, perform class balance')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--filename', type=str, help='Path to the input CSV file')
    parser.add_argument('--from_bucket', action='store_true', help='Get the file from the bucket')
    parser.add_argument('--bucket_key_path', type=str, help='Path to the key file for the bucket')
    parser.add_argument('--n_samples', type=int, help='Number of samples per class (-1 means undersampling to size of least common class)')
    parser.add_argument('--binary', action='store_true', help='Group data into two classes (0, 1)')
    parser.add_argument('--drop_classes', nargs='+', choices=['N', 'F', 'Q', 'S', 'V'], help='List of classes to drop (N, F, Q, S, V)')
    parser.add_argument("--output_dir", type=str, help="Path to save the results")
    parser.add_argument("--scaler_name", type=str, help="Scaler to be used")

    args = parser.parse_args()

    # Define default values
    defaults = {
        'from_bucket': False,
        'bucket_key_path': None,
        'n_samples': -1,
        'binary': False,
        'drop_classes': [],
        'output_dir': os.getcwd(),
        'scaler_name': "MeanVariance"
    }

    # Load config file first
    config = defaults.copy()
    if args.config:
        with open(args.config, 'r') as file:
            config.update(json.load(file))

    # Override config with CLI arguments (only if they're explicitly provided)
    for key, value in vars(args).items():
        if value is not None and key != 'config':  # Override only if CLI argument is provided
            config[key] = value
            print(f"Overriding {key} with {value}")

    print(f"Using final config:\n{config}")
    if args.from_bucket and args.bucket_key_path is None:
        print('Error: bucket_key_path is required when from_bucket is True')
        exit(1)

    if args.from_bucket:
        df = df_from_bucket(args.bucket_key_path)
    else:
        print(f"Reading from {config['filename']}")
        df = pd.read_csv(config['filename'])
    # assign bucket filename to args.filename if using bucket
    if args.from_bucket:
        config['filename'] = 'MIT-BIH.csv'

    df_tr, df_te = preproc(df,
                           n_samples=config['n_samples'], drop_classes=config['drop_classes'],
                           binary=config['binary'], scaler_name=config['scaler_name'])

    # Save to csv
    out_filename = prepare_filename(**config)
    print(f'Saving to {out_filename}_train.csv and {out_filename}_test.csv')
    df_tr.to_csv(f"{out_filename}_train.csv", index=False)
    df_te.to_csv(f"{out_filename}_test.csv", index=False)
