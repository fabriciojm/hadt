import streamlit as st
import requests
import pandas as pd

st.title("Arrhythmia Detection")

models = {"CNN Binary": "cnn_binary_model.h5",
          "LSTM Binary": "lstm_binary_model.h5",
          "PCA XGBoost Binary": "pca_xgboost_binary_model.pkl",
          "CNN Multi": "cnn_multi_model.h5",
          "LSTM Multi": "lstm_multi_model.h5",
          "PCA XGBoost Multi": "pca_xgboost_multi_model.pkl"}

# Model selection
model_name = st.selectbox("Select a Model", list(models.keys()))

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df)

    if st.button("Predict"):
        model = models[model_name]

        # Call the API
        response = requests.post(
            "https://fabriciojm-hadt-api.hf.space/predict/",
            json={"model_name": model, "input_data": df},
        )

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.write(f"Prediction using {model_name}:", prediction)
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
