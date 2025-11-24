import numpy as np
import pickle
from scipy.signal import correlate
from pathlib import Path


class BatchCalibrator:
    def __init__(self, lag=0, scale=1.0):
        self.lag = lag
        self.scale = scale

    def apply(self, x):
        # shift
        if self.lag > 0:
            x_shifted = np.concatenate([np.zeros(self.lag), x[:-self.lag]])
        elif self.lag < 0:
            x_shifted = np.concatenate([x[-self.lag:], np.zeros(-self.lag)])
        else:
            x_shifted = x.copy()

        # scale
        return self.scale * x_shifted


def load_default_calibrator():
    p = Path(__file__).parent / "calibrator.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)
