import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import argparse, os, io
from google.cloud import storage

from tslearn.utils import to_time_series_dataset
from imblearn.over_sampling import SMOTE


def df_from_bucket():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/fabricio/arrhythmia-442911-3fe797ff4111.json'
    storage_client = storage.Client()
    bucket_name = 'arrhythmia_raw_data'
    file_name = 'MIT-BIH_raw.csv'

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download as bytes
    content = blob.download_as_bytes()

    # Convert to pandas DataFrame (for CSV files)
    df = pd.read_csv(io.BytesIO(content))
    return df

def apply_smote(X, y):
    # This function is not working
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)

    # Reshape back to time series format
    X_resampled = X_resampled.reshape(-1, 180)
    X = X.reshape(-1, 180)

    # Convert back to dataframe
    X = pd.DataFrame(X, columns=df.drop(columns='target').columns)
    y = pd.Series(y, name='target')
    X_resampled = pd.DataFrame(X_resampled, columns=df.drop(columns='target').columns)
    y_resampled = pd.Series(y_resampled, name='target')

    # Shuffle order
    rand_idx = np.random.permutation(X_resampled.index)
    X_resampled = X_resampled.loc[rand_idx].reset_index(drop=True)
    y_resampled = y_resampled.loc[rand_idx].reset_index(drop=True)

    return X_resampled, y_resampled

def preproc(filename, n_samples=-1, drop_classes=[], binary=False, smote=False,
            split=False, output_dir=None, scaler_name='Standard'):

    df = pd.read_csv(filename)

    # Dropping redundant index
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Original encoding of the classes
    enc = {'F': 0, 'N': 1, 'Q': 2, 'S': 3, 'V': 4}
    #drop rows where target is in drop_classes
    enc_drop_classes = [enc[c] for c in drop_classes]
    df = df[~df['target'].isin(enc_drop_classes)]

    # number of out classes
    # n_classes_out = df.target.nunique()

    # Have 0 as the positive normal class
    replace = {1: 0, 0: 1}
    df['target'] = df['target'].apply(lambda x: replace[x] if x in replace else x)

    # group data into two classes if binary is True, i.e.1 would group all classes that are not zero
    if binary:
        df['target'] = df['target'].apply(lambda x: 1 if x != 0 else 0)

    # Reshape data for tslearn (samples, timestamps, features)
    X = to_time_series_dataset(df.drop(columns=['target']))
    y = df['target'].values

    # # Scale the time series
    # scaler = TimeSeriesScalerMinMax()
    # X_scaled = scaler.fit_transform(X)

    # Reshape
    X = X.reshape(X.shape[0], -1)

    if smote:
        # X_resampled, y_resampled = apply_smote(X_reshaped, y)
        pass
    # undersample should be a function
    else:
        # if n_samples is -1, undersampling to size of least common class
        if n_samples == -1:
            n_samples = df.target.value_counts().sort_values(ascending=False).iloc[-1]
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

        # should be refactored
        # Convert back to dataframe
        # X = pd.DataFrame(X, columns=df.drop(columns='target').columns)
        # y = pd.Series(y, name='target')

    print('Resulting features shape', X.shape)
    print('Resulting target shape', y.shape)

    # train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    if scaler_name == "Standard":
        scaler = TimeSeriesScalerMeanVariance()
    elif scaler_name == "MinMax":
        scaler = TimeSeriesScalerMinMax()
    else:
        print(f"Error: {scaler_name} not known.")
        return

    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    X_tr = pd.DataFrame(X_tr.squeeze(), columns=df.drop(columns='target').columns)
    y_tr = pd.Series(y_tr, name='target')

    X_te = pd.DataFrame(X_te.squeeze(), columns=df.drop(columns='target').columns)
    y_te = pd.Series(y_te, name='target')

    return X_tr, X_te, y_tr, y_te

    # if split == True:
        # split_data(res, test_size_test=0.2, test_size_val=0.2, out_filename=out_filename)

def prepare_filename(args):
    out_filename = os.path.basename(args.filename)[:-4]
    out_filename = out_filename.replace('raw', 'preproc')
    if args.binary:
        out_filename += '_binary'
    if args.smote:
        out_filename += '_smote'
    out_filename += f"_{args.scaler_name}"
    out_filename += f'_drop{"".join(args.drop_classes)}'
    out_filename = os.path.join(args.output_dir, out_filename)
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
    parser.add_argument('filename', type=str, help='Path to the input CSV file')
    parser.add_argument('--n_samples', type=int, help='Number of samples per class (defaults to -1, which means undersampling to size of least common class)', default=-1)
    parser.add_argument('--binary', action='store_true', help='Group data into two classes (0, 1)', default=False)
    parser.add_argument('--smote', action='store_true', help='Use SMOTE oversampling (to be finished)', default=False)
    # parser.add_argument('--split', action='store_true', help='Start the train_test_val split', default=False)
    parser.add_argument('--drop_classes', nargs='+', choices=['N', 'F', 'Q', 'S', 'V'], help='List of classes to drop (N, F, Q, S, V)', default=[])
    parser.add_argument("--output_dir", type=str, default=os.getcwd(), help="Path to save the results")
    parser.add_argument("--scaler_name", type=str, default="Standard", help="Scaler to be used (default Standard)")

    args = parser.parse_args()

    if args.drop_classes == []:
        print('Preprocess keeping all classes')
    else:
        print(f'Preprocess dropping {args.drop_classes} class(es)')

    if args.binary:
        print("Preparing binary file.")

    print(f"Using scaler {args.scaler_name}")

    X_tr, X_te, y_tr, y_te = preproc(args.filename,
        n_samples=args.n_samples,
        drop_classes=args.drop_classes,
        binary=args.binary,
        smote=args.smote,
        output_dir=args.output_dir,
        scaler_name=args.scaler_name)
            # split=args.split)

    # X_tr, X_te, y_tr, y_te = pandify(X_tr, X_te, y_tr, y_te)

    out_filename = prepare_filename(args)
    # out_filename += '.csv'
    print(f'Saving to {out_filename}.csv')
    res_tr = pd.concat([X_tr, y_tr], axis=1)
    res_te = pd.concat([X_te, y_te], axis=1)
    res_tr.to_csv(f"{out_filename}_train.csv", index=False)
    res_te.to_csv(f"{out_filename}_test.csv", index=False)
