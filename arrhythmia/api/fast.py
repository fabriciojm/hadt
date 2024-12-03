from fastapi import FastAPI
from arrhythmia.ml_logic.binary.cnn import apply_cnn
from arrhythmia.ml_logic.preproc import preproc
import pandas as pd
import numpy as np
from utils_gcp import load_model
from preprocess import preproc

app = FastAPI()
app.state.model = load_model()

@app.get("/")
def root():
    return dict(greeting="Hello")

@app.post("/predict")
def predict(filename):
    model = app.state.model
    if not model:
        model = app.state.model = load_model()

    # PREPROCESS
    data = pd.read_csv(filename)
    X_proc = preproc(data)

    # PREDICT
    y_pred = model.predict(X_proc)

    # CHECK THE RIGHT FORMAT TO RETURN
    return y_pred
