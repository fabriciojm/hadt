from fastapi import FastAPI, File, UploadFile
from arrhythmia.api.utils_gcp import load_model
from arrhythmia.api.preproc import preproc_xgb_single, label_decoding
import pandas as pd
from io import StringIO

app = FastAPI()
app.state.model = load_model()

@app.get("/")
def root():
    return dict(greeting="Hello")

@app.post("/predict")
async def predict(filepath_csv: UploadFile = File(...)):
    model = app.state.model
    if not model:
        model = app.state.model = load_model()

    file_content = await filepath_csv.read()  # Read the file content as bytes
    X = pd.read_csv(StringIO(file_content.decode('utf-8')))  # Decode bytes and read into pandas DataFrame

    df = preproc_xgb_single(X=X.T, pca_model_path="arrhythmia/api/production/pca.pkl")
    y_pred = model.predict(df)
    y_pred = label_decoding(value=y_pred[0], path="arrhythmia/api/production/label_encoding.pkl")
    return y_pred
