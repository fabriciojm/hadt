import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

def predict_pca_xgboost(X_train, y_train, X_test):
    data_features = X_train.columns

    # scaling to have the data centered around 0 (necessary for PCA)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=data_features)

    # compute the principal components
    pca = PCA()
    pca.fit(X_train)

    W = pca.components_

    W_df = pd.DataFrame(W.T,
                index=data_features,
                columns=[f'PC{i}' for i in data_features])

    # project out dataset into this new set of PCs
    X_proj = pca.transform(X_train)
    X_proj = pd.DataFrame(X_proj,
                columns=[f'PC{i}' for i in data_features])

    # Compute PCs
    eig_vals, eig_vecs = np.linalg.eig(np.dot(X_train.T,X_train))

    W = pd.DataFrame(eig_vecs,
                index=data_features,
                columns=[f'PC{i}' for i in data_features])

    # XGBoost Classifier
    model = XGBClassifier()
    k = 8 # best estimated number of PCs to consider to have around 0.97 of accuracy

    pca_k = PCA(n_components=k).fit(X_train)
    X_proj_k = pd.DataFrame(pca_k.transform(X_train), columns=[f'PC{i}' for i in range(1,k+1)])

    model.fit(X_proj_k, convert(y_train))

    X_test_proj = pd.DataFrame(pca_k.transform(X_test), columns=[f'PC{i}' for i in range(1,k+1)])
    y_pred = invert(model.predict(X_test_proj))

    return y_pred

def convert(y):
    '''function to convert y in the format expected by XGBClassifier'''
    return y.map({
        0: 0,
        2: 1,
        3: 2,
        4: 3
    })

def invert(y):
    '''function to convert back our predictions in the original format'''
    return pd.Series(y).map({
        0: 0,
        1: 2,
        2: 3,
        3: 4
    })

# def main(args):
#     filename_train = args.filename_train
#     filename_test = args.filename_test
#     print(f"Processing files: \n {filename_train} \n {filename_test}")

#     df_train = pd.read_csv(filename_train)
#     df_test = pd.read_csv(filename_test)

#     X_tr = df_train.drop(columns='target')
#     y_tr = df_train['target']
#     X_te = df_test.drop(columns='target')
#     y_te = df_test['target']

#     if args.small_run:
#         X_tr, X_te = X_tr.iloc[:10], X_te.iloc[:10]
#         y_tr, y_te = y_tr[:10], y_te[:10]
#         y_pred = Parallel(n_jobs=args.n_jobs)(
#             delayed(predict_dtw_dtaidistance)(X_tr, y_tr, test_sample)
#             for test_sample in X_te.itertuples()
#         )
#         print(classification_report(y_te, y_pred))
#         print(confusion_matrix(y_te, y_pred))
#         return

#     if args.train_samples is not None:
#         X_tr = X_tr.iloc[:args.train_samples]
#         y_tr = y_tr[:args.train_samples]
#     if args.test_samples is not None:
#         X_te = X_te.iloc[:args.test_samples]
#         y_te = y_te[:args.test_samples]

#     for window in args.windows:
#         print(f"Window size: {window}")
#         y_pred = Parallel(n_jobs=args.n_jobs)(
#             delayed(predict_dtw_dtaidistance)(X_tr, y_tr, test_sample)
#             for test_sample in X_te.itertuples()
#         )
#         print(classification_report(y_te, y_pred))
#         print(confusion_matrix(y_te, y_pred))
#         #save results
#         results = pd.DataFrame({'y_true': y_te, 'y_pred': y_pred})
#         results.to_csv(f"{args.output_dir}/results_dtw_dtaidistance_window_{window}.csv", index=False)
#         print(f"Results saved to {args.output_dir}/results_dtw_dtaidistance_window_{window}.csv")
#     return




# main function
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("filename_train", type=str, help="Path to the preprocessed training data file")
#     parser.add_argument("filename_test", type=str, help="Path to the preprocessed test data file")
#     parser.add_argument("--windows", type=int, default=[10], nargs='+', help="Window size(s) for DTW")
#     parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing")
#     parser.add_argument("--small_run", action="store_true", help="Run a small version of the experiment")
#     parser.add_argument("--train_samples", type=int, default=None, help="Number of training samples to use (none given = use all)")
#     parser.add_argument("--test_samples", type=int, default=None, help="Number of test samples to use (none given = use all)")
#     parser.add_argument("--output_dir", type=str, default=os.getcwd(), help="Path to save the results")
#     args = parser.parse_args()

#     if args.small_run:
#         print("Running small run, overriding window size (to 10) and n_jobs (to 1)")
#         args.windows = [10]
#         args.n_jobs = 1
#     main(args)
