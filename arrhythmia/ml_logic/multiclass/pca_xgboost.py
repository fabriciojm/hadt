import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from arrhythmia.ml_logic.preproc import preproc, label_encoding
from sklearn.metrics import classification_report
import time
import pickle


# def convert(y):
#     '''function to convert y in the format expected by XGBClassifier'''
#     return y.map({
#         0: 0,
#         2: 1,
#         3: 2,
#         4: 3
#     })

# def invert(y):
#     '''function to convert back our predictions in the original format'''
#     return pd.Series(y).map({
#         0: 0,
#         1: 2,
#         2: 3,
#         3: 4
#     })

def pca(X_train, X_test, k):
    # compute the principal components
    pca_k = PCA(n_components=k).fit(X_train)
    # project out dataset into this new set of PCs
    X_train_proj = pd.DataFrame(pca_k.transform(X_train), columns=[f'PC{i}' for i in range(1,k+1)])
    X_test_proj = pd.DataFrame(pca_k.transform(X_test), columns=[f'PC{i}' for i in range(1,k+1)])
    print("pca done!")
    file_pca = "/home/fabricio/pca_multiclass.pkl"
    with open(file_pca, "wb") as file:
        pickle.dump(pca_k, file)
    print('PCA file saved in ', file_pca)
    return X_train_proj, X_test_proj

def fit_xgboost(X_train, y_train):
    # XGBoost Classifier
    model = XGBClassifier()
    # model.fit(X_train, convert(y_train))
    model.fit(X_train, y_train)
    print("fit done!")
    # return trained model
    return model

def predict_xgboost(model, X_test):
    # y_pred = invert(model.predict(X_test))
    y_pred = model.predict(X_test)
    return y_pred

def main(X_train, X_test, y_train, k=10):
    # PCA to reduce the nber of columns for X_train and X_test
    # k = 10 is the best estimated number of PCs to have around 0.95 of accuracy
    X_train_proj, X_test_proj = pca(X_train, X_test, k)
    # fit
    model = fit_xgboost(X_train_proj, y_train)
    # Save the XGBoost model to a .pkl file
    with open("/home/fabricio/pca_xgboost_multiclass.pkl", "wb") as file:
        pickle.dump(model, file)
    # predict
    y_pred = predict_xgboost(model, X_test_proj)
    return y_pred

# main function
if __name__ == "__main__":
    # raw_data = pd.read_csv("../../../raw_data/MIT-BIH.csv")
    raw_data = pd.read_csv("/home/fabricio/arrhythmia_raw_data/MIT-BIH_raw.csv")
    data_train, data_test = preproc(raw_data, drop_classes=["F"], binary=False)
    # data_train = pd.read_csv("../../../raw_data/MIT-BIH_raw_dropF_train.csv")
    # data_train, data_test = label_encoding([data_train, data_test], '/home/chlouette/code/fabriciojm/arrhythmia/arrhythmia/ml_logic/pickles/pca_xgboost_multiclass_label_encoding.pkl')
    data_train, data_test = label_encoding([data_train, data_test], '/home/fabricio/pca_xgboost_multiclass_label_encoding.pkl')
    X_train = data_train.drop(columns="target")
    y_train = data_train.target
    # X_test = pd.read_csv("../../../raw_data/MIT-BIH_raw_dropF_test.csv").drop(columns="target")
    X_test = data_test.drop(columns="target")
    y_test = data_test.target
    t_start = time.time()
    y_pred = main(X_train, X_test, y_train, k=10)
    t_end = time.time()
    print(classification_report(y_test, y_pred))
    print(f"It took {t_end - t_start} seconds for the model to make this prediction.")
