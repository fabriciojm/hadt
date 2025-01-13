from wrappers import LSTMWrapper, XGBWrapper, CNNWrapper
import joblib
from tensorflow.keras.models import load_model


def load_model_by_type(model_path):
    if model_path.suffix == '.h5':
        if 'lstm_multi' in str(model_path):
            return LSTMWrapper(load_model(model_path))
        elif 'cnn_multi' in str(model_path):
            return CNNWrapper(load_model(model_path))
        else:
            raise ValueError("Unsupported model type")
    elif model_path.suffix == '.pkl':
        return XGBWrapper(joblib.load(model_path))
    else:
        raise ValueError("Unsupported model type")

def encoder_from_model(model_name):
    if model_name == "cnn_multi_model.h5":
        return "cnn_multi_label_encoding.pkl"
    elif model_name == "lstm_multi_model.h5":
        return "lstm_multi_label_encoding.pkl"
    elif model_name == "pca_xgboost_multi_model.pkl":
        return "pca_xgboost_multi_label_encoding.pkl"
    elif model_name == "cnn_binary_model.h5":
        return "cnn_binary_label_encoding.pkl"
    elif model_name == "lstm_binary_model.h5":
        return "lstm_binary_label_encoding.pkl"
    elif model_name == "pca_xgboost_binary_model.pkl":
        return "pca_xgboost_binary_label_encoding.pkl"
    else:
        raise ValueError("Unsupported model name")


if __name__ == "__main__":
    from pathlib import Path
    PACKAGE_ROOT = Path(__file__).parent.parent.parent
    MODEL_PATH = PACKAGE_ROOT / "models" / "lstm_multi_model.h5"
    load_model_by_type(MODEL_PATH)