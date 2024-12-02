import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tslearn.utils import to_time_series_dataset
from imblearn.over_sampling import SMOTE

import argparse, os, io, json
from google.cloud import storage

def df_from_bucket(bucket_name='arrhythmia_raw_data', file_name='MIT-BIH_raw.csv', key_path='/home/fabricio/arrhythmia-442911-3fe797ff4111.json'):

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

def apply_smote(X, y):
    pass
    # # This function is not working
    # smote = SMOTE(random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)

    # # Reshape back to time series format
    # X_resampled = X_resampled.reshape(-1, 180)
    # X = X.reshape(-1, 180)

    # # Convert back to dataframe
    # X = pd.DataFrame(X, columns=df.drop(columns='target').columns)
    # y = pd.Series(y, name='target')
    # X_resampled = pd.DataFrame(X_resampled, columns=df.drop(columns='target').columns)
    # y_resampled = pd.Series(y_resampled, name='target')

    # # Shuffle order
    # rand_idx = np.random.permutation(X_resampled.index)
    # X_resampled = X_resampled.loc[rand_idx].reset_index(drop=True)
    # y_resampled = y_resampled.loc[rand_idx].reset_index(drop=True)

    # return X_resampled, y_resampled


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

def preproc(df, n_samples=-1, drop_classes=[], binary=False, smote=False, split=False,
            scaler_name='MeanVariance'):

    # Create a copy to avoid funny errors
    df = df.copy()

    if drop_classes == []:
        print('Preprocess keeping all classes')
    else:
        print(f'Preprocess dropping {drop_classes} class(es)')
    if binary:
        print("Preparing two class file.")

    # Dropping redundant index
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Original encoding of the classes
    enc = {'F': 0, 'N': 1, 'Q': 2, 'S': 3, 'V': 4}
    #drop rows where target is in drop_classes (using loc for proper assignment)
    enc_drop_classes = [enc[c] for c in drop_classes]
    df = df.loc[~df['target'].isin(enc_drop_classes)].reset_index(drop=True)

    # Have 0 as the normal class
    replace = {1: 0, 0: 1}
    df['target'] = df['target'].apply(lambda x: replace[x] if x in replace else x)

    # group data into two classes if binary is True, i.e.1 would group all classes that are not zero
    if binary:
        df['target'] = df['target'].apply(lambda x: 1 if x != 0 else 0)

    # Reshape data for tslearn (samples, timestamps, features)
    X = to_time_series_dataset(df.drop(columns=['target']))
    y = df['target'].values

    # Reshape
    X = X.reshape(X.shape[0], -1)


    # Balance classes
    if smote:
        pass
    else:
        X, y = undersample(X, y, n_samples)

    print('Resulting features shape', X.shape)
    print('Resulting target shape', y.shape)


    # Scale the time series
    if scaler_name == "MeanVariance":
        scaler = TimeSeriesScalerMeanVariance()
    elif scaler_name == "MinMax":
        scaler = TimeSeriesScalerMinMax()
    else:
        print(f"Error: {scaler_name} not known.")
        return
    # No data leakage doing fit_transform before split
    # as tslearn acts on each time series independently
    X = scaler.fit_transform(X)

    # train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    ts_features = df.drop(columns='target').columns
    df_tr = pandasify(X_tr, y_tr, ts_features)
    df_te = pandasify(X_te, y_te, ts_features)

    return df_tr, df_te

    # if split == True:
        # split_data(res, test_size_test=0.2, test_size_val=0.2, out_filename=out_filename)

def prepare_filename(**kwargs):
    # Extract required arguments from kwargs
    filename = kwargs.get("filename", "default.csv")
    binary = kwargs.get("binary", False)
    smote = kwargs.get("smote", False)
    scaler_name = kwargs.get("scaler_name", "Standard")
    drop_classes = kwargs.get("drop_classes", [])
    output_dir = kwargs.get("output_dir", "./")

    # Prepare the filename
    out_filename = os.path.basename(filename)[:-4]
    out_filename = out_filename.replace('raw', 'preproc')

    if binary:
        out_filename += '_binary'
    if smote:
        out_filename += '_smote'
    out_filename += f'_{scaler_name}'
    if drop_classes:
        out_filename += f"_drop{''.join(drop_classes)}"
    out_filename = os.path.join(output_dir, out_filename)

    return out_filename


# def split_data(df, test_size_test=0.2, test_size_val=0.2, out_filename="split-data"):
#     X = df.drop(columns="target")
#     y = df.target
#     # train_test split
#     X_train_temp, X_test, y_train_temp, y_test = train_test_split(
#         X, y,
#         test_size=test_size_test,
#         random_state=42
#     )
#     # train_validation_test split
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_temp, y_train_temp,
#         test_size=test_size_val,
#         random_state=42
#     )


#     # convert results in csv files
#     ## train
#     res_train = pd.concat([X_train, y_train], axis=1)
#     print(f"Saving to {out_filename}_train.csv")
#     res_train.to_csv(f"{out_filename}_train.csv", index=False)
#     ## test
#     res_test = pd.concat([X_test, y_test], axis=1)
#     print(f"Saving to {out_filename}_test.csv")
#     res_test.to_csv(f"{out_filename}_test.csv", index=False)
#     ## val
#     res_val = pd.concat([X_val, y_val], axis=1)
#     print(f"Saving to {out_filename}_val.csv")
#     res_val.to_csv(f"{out_filename}_val.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the data, perform class balance')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--filename', type=str, help='Path to the input CSV file')
    parser.add_argument('--from_bucket', action='store_true', help='Get the file from the bucket', default=False)
    parser.add_argument('--n_samples', type=int, help='Number of samples per class (defaults to -1, which means undersampling to size of least common class)', default=-1)
    parser.add_argument('--binary', action='store_true', help='Group data into two classes (0, 1)', default=False)
    parser.add_argument('--smote', action='store_true', help='Use SMOTE oversampling (to be finished)', default=False)
    parser.add_argument('--split', action='store_true', help='Start the train_test_val split', default=False)
    parser.add_argument('--drop_classes', nargs='+', choices=['N', 'F', 'Q', 'S', 'V'], help='List of classes to drop (N, F, Q, S, V)', default=[])
    parser.add_argument("--output_dir", type=str, default=os.getcwd(), help="Path to save the results")
    parser.add_argument("--scaler_name", type=str, default="MeanVariance", help="Scaler to be used (default MeanVariance)")

    args = parser.parse_args()

    # Load config file
    config = {}
    if args.config:
        with open(args.config, 'r') as file:
            config = json.load(file)

    # Override config with CLI arguments
    for key, value in vars(args).items():
        if value is not None:  # Override if CLI argument is provided
            config[key] = value

    # print(f"Using scaler {args.scaler_name}")

    # X_tr, X_te, y_tr, y_te = preproc(args.filename,
    #     n_samples=args.n_samples,
    #     drop_classes=args.drop_classes,
    #     binary=args.binary,
    #     smote=args.smote,
    #     output_dir=args.output_dir,
    #     scaler_name=args.scaler_name)
    #         # split=args.split)

    print(f"Using final config:\n{config}")
    df = df_from_bucket() if args.from_bucket else pd.read_csv(args.filename)
    df_tr, df_te = preproc(df,
                           n_samples=config['n_samples'], drop_classes=config['drop_classes'],
                           binary=config['binary'], smote=config['smote'],
                           scaler_name=config['scaler_name'], split=config['split'])

    # Save to csv
    # out_filename = prepare_filename(args)
    out_filename = prepare_filename(**config)
    # out_filename += '.csv'
    print(f'Saving to {out_filename}_train.csv and {out_filename}_test.csv')
    df_tr.to_csv(f"{out_filename}_train.csv", index=False)
    df_te.to_csv(f"{out_filename}_test.csv", index=False)
