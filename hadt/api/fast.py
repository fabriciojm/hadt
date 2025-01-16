from fastapi import FastAPI, File, UploadFile, HTTPException
from huggingface_hub import hf_hub_download

from utils import load_model_by_type, encoder_from_model
from preproc import label_decoding
import pandas as pd
from io import StringIO
from pathlib import Path
import os

# Get the absolute path to the package directory
PACKAGE_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PACKAGE_ROOT / "models"

app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Dynamically set the cache directory
DEFAULT_CACHE_DIR = "./cache"  # Local directory for cache
CACHE_DIR = os.getenv("CACHE_DIR", DEFAULT_CACHE_DIR)

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


# Use absolute paths with Path objects
model_cache = {}
encoder_cache = {}
HF_REPO_ID = "fabriciojm/hadt-models"

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
        # print("model_name", model_name)
        # print("model_path", model_path)
        try:
            if not model_path.exists():
                # Convert downloaded paths to Path objects
                model_path = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=f"{model_name}", cache_dir=CACHE_DIR))
                encoder_path = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=f"{encoder_name}", cache_dir=CACHE_DIR))
            # print("model_path", model_path)
            model_cache[model_name] = load_model_by_type(model_path)  # Ensure string path for loading
            encoder_cache[model_name] = encoder_path
        except Exception as e:
            print(f"Error loading model: {str(e)}")  # Add debug print
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found: {str(e)}")
    model = app.state.model = model_cache[model_name]

    # Read the uploaded CSV file
    file_content = await filepath_csv.read()
    X = pd.read_csv(StringIO(file_content.decode('utf-8')), header=None)
    y_pred = model.predict_with_pipeline(X)
    
    # Decode prediction using absolute path
    
    y_pred = label_decoding(value=y_pred[0], path=encoder_cache[model_name])
    
    return {"prediction": y_pred}
