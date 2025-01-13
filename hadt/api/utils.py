# from google.cloud import storage
# from hadt.api.params import *
from hadt.api.wrappers import LSTMWrapper, XGBWrapper, CNNWrapper
import joblib
from tensorflow.keras.models import load_model

from pathlib import Path

# Get the absolute path to the package directory
PACKAGE_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PACKAGE_ROOT / "models" / "lstm_multi_model.h5"


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

if __name__ == "__main__":
    load_model_by_type(MODEL_PATH)