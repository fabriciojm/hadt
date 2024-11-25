import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.utils import resample

import sys
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier

from tslearn.utils import to_time_series_dataset
from imblearn.over_sampling import SMOTE


def preproc_smote(filename: str, n_samples=-1):
    df = pd.read_csv(filename)
    # df = pd.read_csv('MIT-BIH_preproc.csv')
    # Dropping redundant index
    np.all(df.index == df['Unnamed: 0'])
    df.drop(columns=['Unnamed: 0'], inplace=True)
    # Have 0 as the positive normal class
    replace = {1: 0, 0: 1}
    df['target'] = df['target'].apply(lambda x: replace[x] if x in replace else x)

    # Check consistent size should be done at the beginning of function, to be fixed
    if n_samples > 0:
        n_max = df.target.value_counts().sort_values(ascending=False).iloc[0]
        n_min = df.target.value_counts().sort_values(ascending=False).iloc[1]
        if n_samples > n_max or n_samples < n_min:
            print(f'Value counts error, n_samples {n_samples} should be within first two values')
            print(df.target.value_counts())
            return
        idx_0 = df[df.target == 0].index[:n_samples]
        idx_1 = df[df.target != 0].index
        df = pd.concat([df.iloc[idx_0], df.iloc[idx_1]], axis=0)
        print('df has been reshaped to ', df.shape)

    # return

    # print(df.target.value_counts())
    # print(df.target.value_counts(normalize=True))

    # df_N = df[df.target == 0].sample(20).drop(columns=['target'])

    # fig, ax = plt.subplots(figsize=(10, 6))
    # for _, row in df_N.iterrows():
    #     ax.plot(row.values, alpha=0.3)
    # ax.set_title('Multiple Normal Beats')
    # ax.set_ylim(-4, 8)
    # plt.show()

    # For the record: encoding the classes
    enc = {'N': 0, 'F': 1, 'Q': 2, 'S': 3, 'V': 4}
    # reverse the encoding
    # dec = {v: k for k, v in enc.items()}


    # # do the same for the other classes
    # dfs = [df[df.target == i].sample(20).drop(columns=['target'])
    #        for i in range(1, 5)]

    # _, ax = plt.subplots(1, 4, figsize=(24, 6))

    # for i, df_class in enumerate(dfs):
    #     for _, row in df_class.iterrows():
    #         ax[i].plot(row.values, alpha=0.3)
    #     ax[i].set_title(f'Class {dec[i+1]}')
    #     ax[i].set_ylim(-4, 8)
    # plt.show()

    ## SMOTE ##

    # Reshape data for tslearn (samples, timestamps, features)
    X = to_time_series_dataset(df.drop(columns=['target']))
    y = df['target'].values

    # Scale the time series
    scaler = TimeSeriesScalerMinMax()
    X_scaled = scaler.fit_transform(X)

    # Reshape for SMOTE (samples, timestamps*features)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], -1)

    # Apply SMOTE
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

    print('Resulting features shape', X_resampled.shape)
    out_filename = filename[:-4] + '_smote.csv'
    print(f'Saving to {out_filename}')
    pd.concat([X_resampled, y_resampled], axis=1).to_csv(out_filename)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preproc_smote.py filename n_samples ")
        sys.exit(1)

    filename = sys.argv[1]
    n_samples = int(sys.argv[2])
    preproc_smote(filename, n_samples=n_samples)
