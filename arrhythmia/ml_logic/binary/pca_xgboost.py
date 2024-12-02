import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from arrhythmia.ml_logic.preproc import preproc
from sklearn.metrics import classification_report

def pca(X_train, X_test, k):
    # compute the principal components
    pca_k = PCA(n_components=k).fit(X_train)
    # project out dataset into this new set of PCs
    X_train_proj = pd.DataFrame(pca_k.transform(X_train), columns=[f'PC{i}' for i in range(1,k+1)])
    X_test_proj = pd.DataFrame(pca_k.transform(X_test), columns=[f'PC{i}' for i in range(1,k+1)])
    print("pca done!")
    return X_train_proj, X_test_proj

def fit_xgboost(X_train, y_train):
    # XGBoost Classifier
    model = XGBClassifier()
    model.fit(X_train, y_train)
    print("fit done!")
    # return trained model
    return model

def predict_xgboost(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def main(X_train, X_test, y_train, k=8):
    # PCA to reduce the nber of columns for X_train and X_test
    # k = 8 is the best estimated number of PCs to have around 0.97 of accuracy
    X_train_proj, X_test_proj = pca(X_train, X_test, k)
    # fit
    model = fit_xgboost(X_train_proj, y_train)
    # predict
    y_pred = predict_xgboost(model, X_test_proj)
    return y_pred

# main function
if __name__ == "__main__":
    raw_data = pd.read_csv("../../../raw_data/MIT-BIH.csv")
    data_train, data_test = preproc(raw_data, drop_classes=["F"], n_samples=10000, binary=True)
    # data_train = pd.read_csv("../../../raw_data/MIT-BIH_raw_binary_dropF_train.csv")
    X_train = data_train.drop(columns="target")
    y_train = data_train.target
    # X_test = pd.read_csv("../../../raw_data/MIT-BIH_raw_binary_dropF_test.csv").drop(columns="target")
    X_test = data_test.drop(columns="target")
    y_test = data_test.target
    y_pred = main(X_train, X_test, y_train, k=8)
    print(classification_report(y_test, y_pred))
