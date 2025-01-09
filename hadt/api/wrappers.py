from tensorflow.keras.models import load_model
import numpy as np
import joblib

class TensorFlowModelWrapper:
    def __init__(self, model_path):
        # Load TensorFlow model
        self.model = load_model(model_path)

    def predict(self, data):
        return self.model.predict(data).tolist()


class SklearnModelWrapper:
    def __init__(self, model_path):
        # Load Scikit-learn model
        self.model = joblib.load(model_path)

    def predict(self, data):
        return self.model.predict(data).tolist()
