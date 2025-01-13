from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import pickle

def preproc_single(X):
    # to be called in inference/api
    in_shape = X.shape
    if X.shape != (1, 180):
        print('File shape is not (1, 180) but ', in_shape)

    X = to_time_series_dataset(X)
    X = X.reshape(in_shape[0], -1)
    scaler = TimeSeriesScalerMeanVariance()
    X = scaler.fit_transform(X)
    return X.reshape(in_shape)

def label_decoding(value, path):
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    inverse_mapping = {v: k for k, v in mapping.items()}
    return inverse_mapping[value]