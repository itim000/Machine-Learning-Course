import numpy as np

def to_features(X):
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    return X