import numpy as np

def log_transform(x):
    return np.sign(x) * np.log1p(np.abs(x))