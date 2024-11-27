import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.utils import resample

import argparse

from tslearn.utils import to_time_series_dataset
from imblearn.over_sampling import SMOTE

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

def preproc(filename, n_samples=-1, drop_classes=[], binary=False, smote=False):
    df = pd.read_csv(filename)

    # Dropping redundant index
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Original encoding of the classes
    enc = {'F': 0, 'N': 1, 'Q': 2, 'S': 3, 'V': 4}
    #drop rows where target is in drop_classes
    enc_drop_classes = [enc[c] for c in drop_classes]
    df = df[~df['target'].isin(enc_drop_classes)]

    # number of out classes
    n_classes_out = df.target.nunique()

    # Have 0 as the positive normal class
    replace = {1: 0, 0: 1}
    df['target'] = df['target'].apply(lambda x: replace[x] if x in replace else x)

    # group data into two classes if binary is True, i.e.1 would group all classes that are not zero
    if binary:
        df['target'] = df['target'].apply(lambda x: 1 if x != 0 else 0)

    # Reshape data for tslearn (samples, timestamps, features)
    X = to_time_series_dataset(df.drop(columns=['target']))
    y = df['target'].values

    # Scale the time series
    scaler = TimeSeriesScalerMinMax()
    X_scaled = scaler.fit_transform(X)

    # Reshape for SMOTE (samples, timestamps*features)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], -1)

    if smote:
        X_resampled, y_resampled = apply_smote(X_reshaped, y)

    else:
        # if n_samples is -1, undersampling to size of least common class
        if n_samples == -1:
            n_samples = df.target.value_counts().sort_values(ascending=False).iloc[1]
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
        X_resampled = X_reshaped[selected_indices]
        y_resampled = y[selected_indices]


        # should be refactored
        # Convert back to dataframe
        # X = pd.DataFrame(X, columns=df.drop(columns='target').columns)
        # y = pd.Series(y, name='target')
        X_resampled = pd.DataFrame(X_resampled, columns=df.drop(columns='target').columns)
        y_resampled = pd.Series(y_resampled, name='target')

    print('Resulting features shape', X_resampled.shape)
    print('Resulting target shape', y_resampled.shape)
    #out_filename reflects if binary, if smote, and if drop_classes
    out_filename = filename[:-4]
    if binary:
        out_filename += '_binary'
    if smote:
        out_filename += '_smote'
    out_filename += f'_{n_classes_out}classes'
    out_filename += '.csv'
    print(f'Saving to {out_filename}')
    res = pd.concat([X_resampled, y_resampled], axis=1)
    res.to_csv(out_filename, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the data, perform class balance using SMOTE')
    parser.add_argument('filename', type=str, help='Path to the input CSV file')
    # n_samples defaults to -1, which means undersampling and smote is False
    # in this case, the number of samples is the number of samples of the least common class
    parser.add_argument('--n_samples', type=int, help='Number of samples per class (defaults to -1, which means undersampling to size of least common class)', default=-1)
    parser.add_argument('--binary', action='store_true', help='Group data into two classes (0, 1)', default=False)
    parser.add_argument('--smote', action='store_true', help='Use SMOTE instead of undersampling', default=False)
    parser.add_argument('--drop_classes', nargs='+', choices=['N', 'F', 'Q', 'S', 'V'], help='List of classes to drop (N, F, Q, S, V)', default=[])
    # parser.add_argument('--classes', type=list, help='Number of classes to keep', default=-1)
    args = parser.parse_args()

    if args.drop_classes == []:
        print('Preprocess keeping all classes')
    else:
        print(f'Preprocess dropping {args.drop_classes} classes')

    preproc(args.filename,
            n_samples=args.n_samples,
            drop_classes=args.drop_classes,
            binary=args.binary,
            smote=args.smote)
