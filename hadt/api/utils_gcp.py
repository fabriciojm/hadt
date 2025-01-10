from google.cloud import storage
from hadt.api.params import *
from hadt.api.wrappers import TensorFlowModelWrapper, SklearnModelWrapper


def load_model(model_path):
    """Load the local model file"""
    # model path is a posix path
    if model_path.suffix == '.h5':
        return TensorFlowModelWrapper(model_path)
    elif model_path.suffix == '.pkl':
        return SklearnModelWrapper(model_path)
    else:
        raise ValueError("Unsupported model file type. Please provide a .h5 or .pkl file.")

if __name__ == "__main__":
    load_model()
