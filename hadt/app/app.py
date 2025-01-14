import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
st.title("Arrhythmia Detection")

models = {
    "LSTM Multi": "lstm_multi_model.h5",
    "CNN Multi": "cnn_multi_model.h5",
    "PCA XGBoost Multi": "pca_xgboost_multi_model.pkl",
    "LSTM Binary": "lstm_binary_model.h5",
    "CNN Binary": "cnn_binary_model.h5",
    "PCA XGBoost Binary": "pca_xgboost_binary_model.pkl",
    }

# Model selection
model_name = st.selectbox("Select a Model", list(models.keys()))

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # st.write("Uploaded Data:", df)

    st.write("Visualized Data:")
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(ax=ax)
    st.pyplot(fig)

    if st.button("Predict"):
        model = models[model_name]

        # Reset the file pointer to the beginning
        uploaded_file.seek(0)

        # Call the API with the file directly
        response = requests.post(
            f"https://fabriciojm-hadt-api.hf.space/predict?model_name={model}",
            files={"filepath_csv": (uploaded_file.name, uploaded_file, "text/csv")},
        )

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.write(f"Prediction using {model_name}:", prediction)
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
