from fastapi import FastAPI, File, UploadFile
from hadt.api.utils_gcp import load_model
# from hadt.api.preproc import preproc_single, label_decoding
from hadt.ml_logic.preproc import preproc_single, label_decoding
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path

# Get the absolute path to the package directory
PACKAGE_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PACKAGE_ROOT / "models"

app = FastAPI()

# Use absolute paths with Path objects
MODEL_PATH = MODEL_DIR / "lstm_multi_model.h5"
LABEL_ENCODER_PATH = MODEL_DIR / "lstm_multi_label_encoding.pkl"

app.state.model = None  # Initialize as None, load on first request

@app.get("/")
def root():
    return dict(greeting="Hello")

@app.post("/predict")
async def predict(filepath_csv: UploadFile = File(...)):
    # Load model if not already loaded
    model = app.state.model
    if not model:
        model = app.state.model = load_model(MODEL_PATH)

    # Read the uploaded CSV file
    file_content = await filepath_csv.read()
    X = pd.read_csv(StringIO(file_content.decode('utf-8')))
    X = X.T
    # Preprocess the data
    X = preproc_single(X)
    X = np.expand_dims(X, axis=1)
    
    # Make prediction
    y_pred = np.argmax(model.predict(X))
    
    # Decode prediction using absolute path
    y_pred = label_decoding(
        value=y_pred, 
        path=LABEL_ENCODER_PATH
    )
    
    return {"prediction": y_pred}
