from fastapi import FastAPI, File, UploadFile, HTTPException
from huggingface_hub import hf_hub_download

from utils import load_model_by_type, encoder_from_model
from preproc import label_decoding
import pandas as pd, numpy as np
from io import StringIO
from pathlib import Path
import joblib
# Get the absolute path to the package directory
PACKAGE_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PACKAGE_ROOT / "models"

app = FastAPI()

# Use absolute paths with Path objects
# MODEL_PATH = MODEL_DIR / "pca_xgboost_multi_model.pkl"
# LABEL_ENCODER_PATH = MODEL_DIR / "pca_xgboost_multi_label_encoding.pkl"
model_cache = {}
encoder_cache = {}
# HF_REPO_ID = "your-username/your-model-repo"

app.state.model = None  # Initialize as None, load on first request

@app.get("/")
def root():
    return dict(greeting="Hello")

@app.post("/predict")
async def predict(model_name: str, filepath_csv: UploadFile = File(...)):
    # Load model if not already loaded
    model_path = MODEL_DIR / f"{model_name}"
    encoder_name = encoder_from_model(model_name)
    encoder_path = MODEL_DIR / encoder_name

    # if model in model_path, load it, otherwise download it from HF
    if model_name not in model_cache:
        try:
            if not model_path.exists():
                model_path = hf_hub_download(repo_id=model_name, filename=f"{model_name}")
                encoder_path = hf_hub_download(repo_id=model_name, filename=f"{encoder_name}")
            model_cache[model_name] = load_model_by_type(model_path)
            encoder_cache[model_name] = encoder_path
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    model = model_cache[model_name]

    # Read the uploaded CSV file
    file_content = await filepath_csv.read()
    X = pd.read_csv(StringIO(file_content.decode('utf-8')), header=None).T
    y_pred = model.predict_with_pipeline(X)
    
    # Decode prediction using absolute path
    
    y_pred = label_decoding(value=y_pred[0], path=encoder_cache[model_name])
    
    return {"prediction": y_pred}
