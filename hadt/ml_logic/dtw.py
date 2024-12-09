import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, confusion_matrix
from dtaidistance.dtw import distance_fast
import argparse, os


def predict_dtw_dtaidistance(X_tr, y_tr, test_sample, window=10):
    # Fixes "non-writable" error
    X_tr = X_tr.to_numpy().copy()
    y_tr = y_tr.to_numpy().copy()

    # itertuples also takes index, remove it
    test_sample = np.array(test_sample[1:]).copy()

    distances = []
    for i in range(len(X_tr)):
        train_sample = X_tr[i].copy()  # Ensure train_sample is writable

        # Compute DTW distance
        distance = distance_fast(test_sample, train_sample, window=window)
        distances.append(distance)

    # Find the nearest neighbor
    nearest_idx = np.argmin(distances)
    return y_tr[nearest_idx]  # Predicted class


def main(args):
    filename_train = args.filename_train
    filename_test = args.filename_test
    print(f"Processing files: \n {filename_train} \n {filename_test}")

    df_train = pd.read_csv(filename_train)
    df_test = pd.read_csv(filename_test)

    X_tr = df_train.drop(columns='target')
    y_tr = df_train['target']
    X_te = df_test.drop(columns='target')
    y_te = df_test['target']

    if args.small_run:
        X_tr, X_te = X_tr.iloc[:10], X_te.iloc[:10]
        y_tr, y_te = y_tr[:10], y_te[:10]
        y_pred = Parallel(n_jobs=args.n_jobs)(
            delayed(predict_dtw_dtaidistance)(X_tr, y_tr, test_sample)
            for test_sample in X_te.itertuples()
        )
        print(classification_report(y_te, y_pred))
        print(confusion_matrix(y_te, y_pred))
        return

    if args.train_samples is not None:
        X_tr = X_tr.iloc[:args.train_samples]
        y_tr = y_tr[:args.train_samples]
    if args.test_samples is not None:
        X_te = X_te.iloc[:args.test_samples]
        y_te = y_te[:args.test_samples]

    for window in args.windows:
        print(f"Window size: {window}")
        y_pred = Parallel(n_jobs=args.n_jobs)(
            delayed(predict_dtw_dtaidistance)(X_tr, y_tr, test_sample)
            for test_sample in X_te.itertuples()
        )
        print(classification_report(y_te, y_pred, digits=4))
        print(confusion_matrix(y_te, y_pred))
        #save results
        results = pd.DataFrame({'y_true': y_te, 'y_pred': y_pred})
        results.to_csv(f"{args.output_dir}/results_dtw_dtaidistance_window_{window}.csv", index=False)
        print(f"Results saved to {args.output_dir}/results_dtw_dtaidistance_window_{window}.csv")
    return




# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename_train", type=str, help="Path to the preprocessed training data file")
    parser.add_argument("filename_test", type=str, help="Path to the preprocessed test data file")
    parser.add_argument("--windows", type=int, default=[10], nargs='+', help="Window size(s) for DTW")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing")
    parser.add_argument("--small_run", action="store_true", help="Run a small version of the experiment")
    parser.add_argument("--train_samples", type=int, default=None, help="Number of training samples to use (none given = use all)")
    parser.add_argument("--test_samples", type=int, default=None, help="Number of test samples to use (none given = use all)")
    parser.add_argument("--output_dir", type=str, default=os.getcwd(), help="Path to save the results")
    args = parser.parse_args()

    if args.small_run:
        print("Running small run, overriding window size (to 10) and n_jobs (to 1)")
        args.windows = [10]
        args.n_jobs = 1
    main(args)
