from fastapi import FastAPI
from arrhythmia.ml_logic.binary.cnn import apply_cnn
from arrhythmia.ml_logic.preproc import preproc
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from arrhythmia.ml_logic.binary.cnn import initialize_model
from tensorflow.keras.callbacks import EarlyStopping

app = FastAPI()

@app.get("/")
def root():
    return dict(greeting="Hello")

# app.state.model.predict()
# app.state.model = apply_cnn()

# @app.get("/predict")
# # Let's use our model
def predict(filename):
    data = pd.read_csv(filename)
    X,y = data.drop(columns=['Unnamed: 0','target']),data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    #adapt dimensions for the Con1D layer
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    #Initialize the model
    model = initialize_model(X_train)

    #fit the model
    es = EarlyStopping(patience=5, restore_best_weights=True)
    print('je suis en train de fiter, ça peut prendre 2min')

    model.fit(X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_split=0.2,
                callbacks=[es],
                verbose =0)

    model.predict

    model = app.state.model
    # assert model is not None
    # return (wavetype=float(y_pred))
