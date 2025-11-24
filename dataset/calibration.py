from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

def load_default_calibrator():
    p = Path(__file__).parent / "calibrator.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)
        
def to_1d(x):
    x = np.array(x, dtype=float)
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError(f"Signal must be 1-D, got {x.shape}")
    return x


def fastdtw_warp(train_signal, test_signal):
    """
    Warp test_signal to align with train_signal.
    Both signals must be 1-D before calling.
    FastDTW requires (T,1) shaped sequences.
    """

    # ensure 1-D
    train_signal = to_1d(train_signal)
    test_signal  = to_1d(test_signal)

    # reshape to (T,1) for FastDTW
    train_vec = train_signal.reshape(-1, 1)
    test_vec  = test_signal.reshape(-1, 1)

    _, path = fastdtw(test_vec, train_vec, dist=euclidean)

    warped = np.zeros(train_signal.shape[0], float)
    counts = np.zeros(train_signal.shape[0], float)

    for i_test, i_train in path:
        warped[i_train] += test_signal[i_test]
        counts[i_train] += 1

    warped[counts > 0] /= counts[counts > 0]
    warped[counts == 0] = train_signal[counts == 0]

    return warped


def scale_to_train(train_signal, warped_signal):
    train_signal = to_1d(train_signal)
    warped_signal = to_1d(warped_signal)

    num = np.dot(train_signal, warped_signal)
    den = np.dot(warped_signal, warped_signal)
    scale = num / den
    return scale * warped_signal, scale


class FastDTWCalibratorMulti:
    def __init__(self):
        self.train_template = None
        self.scale_mean = 1.0

    def fit(self, pairs):
        """
        pairs: list of (train_signal, test_signal)
        Builds a median train template and averaged scaling.
        """

        warped_list = []
        train_list = []
        scale_list = []

        for train_signal, test_signal in pairs:

            # enforce 1-D
            train_signal = to_1d(train_signal)
            test_signal  = to_1d(test_signal)

            # warp using FastDTW
            warped = fastdtw_warp(train_signal, test_signal)

            # scale warped version to match train
            warped_scaled, scale = scale_to_train(train_signal, warped)

            warped_list.append(to_1d(warped_scaled))
            train_list.append(train_signal)
            scale_list.append(scale)

        # median template across all train signals
        self.train_template = np.median(np.vstack(train_list), axis=0)

        # average scale across all pairs
        self.scale_mean = float(np.mean(scale_list))

    def calibrate(self, x):
        """
        Warp and scale a new test pulse to match the learned template.
        """
        x = to_1d(x)

        warped = fastdtw_warp(self.train_template, x)
        calibrated = self.scale_mean * warped
        return calibrated

