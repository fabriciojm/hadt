import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from io import BytesIO

st.title("Heart Arrhythmia Detection Tools (hadt)")

st.markdown("""
This is a demo of the Heart Arrhythmia Detection Tools (hadt) project.
The project is available on [GitHub](https://github.com/fabriciojm/hadt).
""")

models = {
    "LSTM Multiclass": "lstm_multi_model.h5",
    "CNN Multiclass": "cnn_multi_model.h5",
    "PCA XGBoost Multiclass": "pca_xgboost_multi_model.pkl",
    "LSTM Binary": "lstm_binary_model.h5",
    "CNN Binary": "cnn_binary_model.h5",
    "PCA XGBoost Binary": "pca_xgboost_binary_model.pkl",
}

beat_labels = {
    "N": "Normal",
    "Q": "Unknown Beat",
    "S": "Supraventricular Ectopic", 
    "V": "Ventricular Ectopic",
    "A": "Abnormal",
}

# Model selection
classification = ["Multiclass", "Binary"]
model_list = ["LSTM", "CNN", "PCA XGBoost"]

model_selected = st.selectbox("Select a Model", model_list)
classification_selected = st.selectbox("Classification type", classification)

model_name = f"{model_selected} {classification_selected}"

st.markdown("""Upload a CSV file with single heartbeat (csv with 180 points) or load from available examples 
""")

# Option to upload or load a file
option = st.radio("Choose input method", ("Load example file", "Upload CSV file"))

if option == "Load example file":
    # Load example files from Hugging Face dataset
    example_files = ["single_N.csv", "single_Q.csv", "single_S.csv", "single_V.csv"]
    example_selected = st.selectbox("Select an example file", example_files)

    # Load the selected example file
    file_path = hf_hub_download(repo_id='fabriciojm/ecg-examples', repo_type='dataset', filename=example_selected)
    with open(file_path, 'rb') as f:
        file_content = f.read()
    uploaded_file = BytesIO(file_content)
    uploaded_file.name = example_selected  # Set a name attribute to mimic the uploaded file
    df = pd.read_csv(uploaded_file)
    # st.write("Loaded Data:", df)

else:
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # st.write("Uploaded Data:", df)

# Visualize data
if 'df' in locals():
    st.write("Visualized Data:")
    fig, ax = plt.subplots(figsize=(10, 6))
    df.iloc[0].plot(ax=ax)
    st.pyplot(fig)

    if st.button("Predict"):
        model = models[model_name]

        # Reset the file pointer to the beginning
        uploaded_file.seek(0)

        # Call the API with the file directly
        response = requests.post(
            f"https://fabriciojm-hadt-api.hf.space/predict?model_name={model}",
            files={"filepath_csv": (uploaded_file.name, uploaded_file, "text/csv")}
        )

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.write(f"Prediction using {model_name}: {beat_labels[prediction]} (class {prediction}) heartbeat")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
