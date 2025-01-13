import numpy as np
from preproc import preproc_single

class BaseModelWrapper:
    def __init__(self, model):
        self.model = model

    def preprocess(self, data):
        """Default preprocessing (can be overridden)."""
        return preproc_single(data)

    def predict(self, data):
        """Call the model's prediction."""
        raise NotImplementedError("Subclasses must implement predict()")

    def postprocess(self, prediction):
        """Default postprocessing (can be overridden)."""
        return prediction

    def predict_with_pipeline(self, data):
        """Unified prediction pipeline."""
        processed_data = self.preprocess(data)
        raw_prediction = self.predict(processed_data)
        final_output = self.postprocess(raw_prediction)
        return final_output


class LSTMWrapper(BaseModelWrapper):
    def preprocess(self, data):
        # LSTM requires additional dimension expansion
        data = preproc_single(data)
        return np.expand_dims(data, axis=1)  # Add time-step dimension

    def predict(self, data):
        return self.model.predict(data)

    def postprocess(self, prediction):
        # Assume the output is a probability vector; apply argmax
        return np.argmax(prediction, axis=1).tolist()


class XGBWrapper(BaseModelWrapper):
    def predict(self, data):
        return self.model.predict(data).tolist()

class CNNWrapper(BaseModelWrapper):
    def predict(self, data):
        return self.model.predict(data)
    
    def postprocess(self, prediction):
        return np.argmax(prediction, axis=1).tolist()
