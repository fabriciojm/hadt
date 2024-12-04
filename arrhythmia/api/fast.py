from fastapi import FastAPI
from utils_gcp import load_model
from preprocess import preproc_xgb_single

app = FastAPI()
app.state.model = load_model()

@app.get("/")
def root():
    return dict(greeting="Hello")

@app.post("/predict")
def predict(filepath_csv):
    model = app.state.model
    if not model:
        model = app.state.model = load_model()
    print(model)

    df = preproc_xgb_single(filepath=filepath_csv, pca_model_path="production/pca.pkl")
    y_pred = model.predict(df)

    # decoding using preproc.py decode

    return y_pred
