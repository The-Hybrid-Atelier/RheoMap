from signal_processing import *
from test_data import *
from model.tsne_lib.old_function.validate_data import *
import numpy as np
from collections import defaultdict, Counter
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

filename = "/workspaces/RheoPulse/model/master_dataset/validation_data.bin"
validations = load_validation_data(filename)
mean_phase_shift = -0.30443063667337933
mean_scale = 0.00565277492211214
intercept = 37.92185287883444
coefficients = [0.00000000e+00, 5.54934365e-03, 4.18543570e-10]
params = np.concatenate([
    [mean_phase_shift],  # phase shift
    [intercept],         # intercept
    coefficients         # polynomial coefficients
])


def resample_to_match_length_final(data, target_length):
    """Interpolates or resamples data to match the target length."""
    x_original = np.linspace(0, 1, len(data))
    x_target = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_original, data, kind='linear')
    return interpolator(x_target)

def difference(group1, group2):
    """Calculate the area difference between two signals or groups of signals."""
    # If inputs are single arrays, wrap them in lists
    if isinstance(group1, np.ndarray) and group1.ndim == 1:
        group1 = [group1]
    if isinstance(group2, np.ndarray) and group2.ndim == 1:
        group2 = [group2]

    return np.mean([np.trapz(np.abs(arr1 - arr2)) for arr1 in np.array(group1) for arr2 in np.array(group2)])

def transform_fn_final(pulse: np.ndarray,
                        params: np.ndarray,
                        ref_length: int) -> np.ndarray:
    """
    Transform a single timeseries using the given parameters.
    """
    mean_phase_shift = int(params[0])
    intercept = params[1]
    coefficients = params[2:]

    # Resample and shift
    shifted_pulse = np.roll(resample_to_match_length_final(pulse, ref_length),
                           mean_phase_shift)

    # Apply polynomial transformation
    poly = PolynomialFeatures(degree=len(coefficients)-1)
    X = poly.fit_transform(shifted_pulse.reshape(-1, 1))
    transformed = X @ coefficients + intercept

    return transformed

def calibrate_v3(input_data):
    # Use transform_fn directly
    transformed_signal = transform_fn_final(
        pulse=input_data,
        params=params,
        ref_length=1000
    )
    transformed_signal = segment_pulse(transformed_signal)[0]
    return transformed_signal

# transformed_signal2 = calibrate_v3(honey1['data'])
# x = segment_pulse(transformed_signal2)[0]
# x=resample_to_match_length_final(segment_pulse(transformed_signal2)[0],35)
# plt.plot(x)
# plt.savefig('calibrate_v3.png')