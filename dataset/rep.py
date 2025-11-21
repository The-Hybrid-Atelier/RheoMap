import numpy as np
import pandas as pd
import pywt
from scipy.signal import medfilt

def median_filter(data, kernel_size=5):
    """Apply a median filter to the data."""
    return medfilt(data, kernel_size)

def smooth(pulse, window_size=3):
    """Smooth the pulse using a simple moving average."""
    smoothed = np.convolve(pulse, np.ones(window_size) / window_size, mode='valid')
    # Pad to match original length by repeating edge values
    pad_left = (window_size - 1) // 2
    pad_right = window_size - 1 - pad_left
    return np.pad(smoothed, (pad_left, pad_right), mode='edge')

def wavelet_filter(data, wavelet='db1', level=1):
    """Apply a Wavelet Transform based filter to denoise the data."""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(detail, np.std(detail)) for detail in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def filter_pulse(pulse, median_k=5, wavelet='db1', level=1, window_size=3):
    """
    Apply a sequence of filters to a pulse:
    1. Median filter
    2. Wavelet denoising
    3. Moving average smoothing

    Returns:
        A filtered 1D numpy array (shortened due to moving average).
    """
    # Step 1: Median filter
    pulse_med = median_filter(pulse, kernel_size=median_k)

    # Step 2: Wavelet denoising
    pulse_wave = wavelet_filter(pulse_med, wavelet=wavelet, level=level)
    
    # Match original length in case waverec adds samples
    pulse_wave = pulse_wave[:len(pulse_med)]

    # Step 3: Moving average smoothing
    pulse_smooth = smooth(pulse_wave, window_size=window_size)

    return pulse_smooth

def extract_features(pulse):
    """
    Extract geometric features from a pulse signal.
    Automatically applies filtering (median + wavelet + smoothing) before extraction.
    
    Parameters:
    -----------
    pulse : array-like
        Raw pulse signal
        
    Returns:
    --------
    list : 12 extracted features
        [basevalue, min_retraction_value, max_extrusion_value, equilibrium_value,
         max_retraction_time, max_extrusion_time, equilibrium_time, 
         extrusion_period, equilibrium_period, fluid_release_point_time,
         fluid_release_point_value, fluid_release_point_period]
    """
    # Always apply filtering before feature extraction
    
    # pulse = filter_pulse(pulse) #### PRE FILTERING - SINGLE REP FILTERING
    
    df = pd.DataFrame({"time": np.arange(len(pulse)), "signal": pulse})

    # Feature Extraction
    ## Shape Features
    ### Max Extrusion
    basevalue = df["signal"][0]  # f1
    max_extrusion_idx = df["signal"].idxmax()
    max_extrusion_time = df["time"][max_extrusion_idx]  # f6
    max_extrusion_value = df["signal"][max_extrusion_idx]  # f3
    ### Min Retraction
    min_retraction_idx = df["signal"].idxmin()
    min_retraction_time = df["time"][min_retraction_idx]  # f5
    min_retraction_value = df["signal"][min_retraction_idx]  # f2
    ### Look only at the tail end (after max pressure value)
    tail = df.iloc[max_extrusion_idx:, :]
    ### Equilibrium
    equilibrium_idx = tail["signal"].idxmin()
    equilibrium_value = tail["signal"][equilibrium_idx]  # f4
    equilibrium_time = tail["time"][equilibrium_idx]  # f7
    ### Period features
    extrusion_period = max_extrusion_time - min_retraction_time  # f8
    equilibrium_period = equilibrium_time - max_extrusion_time  # f9
    ### Fluid Release Point
    # Calculate dP/dt for the segment between max_extrusion_idx and equilibrium_idx
    dp_dt_segment = np.diff(pulse[max_extrusion_idx:equilibrium_idx])
    # If the segment is empty or has length less than 1, set default values for fluid_release_point
    if len(dp_dt_segment) < 1:
        fluid_release_point_value = 0
        fluid_release_point_time = 0
        fluid_release_point_period = 0
    else:
        # Find the index of the steepest decrease in pressure (largest negative derivative) within the segment
        fluid_release_point_idx_segment = np.argmin(dp_dt_segment)
        # Adjust the index to get the correct position in the original pulse
        fluid_release_point_idx = max_extrusion_idx + fluid_release_point_idx_segment
        # Get the fluid_release_point_value
        fluid_release_point_value = pulse[fluid_release_point_idx]
        # Use the local time variable for each pulse
        fluid_release_point_time = df["time"].iloc[fluid_release_point_idx]
        fluid_release_point_period = fluid_release_point_time - max_extrusion_time

    # Add the extracted features to the feature_data list
    resp = {
        "names": ["basevalue", "min_retraction_value", "max_extrusion_value", "equilibrium_value", "max_retraction_time",
                    "max_extrusion_time", "equilibrium_time", "extrusion_period", "equilibrium_period", "fluid_release_point_time",
                    "fluid_release_point_value", "fluid_release_point_period"],
        "values": [basevalue, min_retraction_value, max_extrusion_value, equilibrium_value, min_retraction_time, max_extrusion_time,
                    equilibrium_time, extrusion_period, equilibrium_period, fluid_release_point_time, fluid_release_point_value,
                    fluid_release_point_period]
    }
    # return resp
    return resp["values"]

def extract_features_nocv(stack, set="vuong"):
    """
    Extracts a feature vector from a vuong_sv_stack (k x n).

    Parameters:
    - stack: np.ndarray of shape (k, n)
    - set: 'vuong' or 'matnoise' to determine which feature set to extract

    Returns:
    - feature_vector: np.ndarray of shape depending on the set
    """
    if set == "vuong":
        means = np.mean(stack, axis=0)
        features = [means]


    elif set == "matnoise":
        means = np.mean(stack, axis=0)
        stds = np.std(stack, axis=0)
        medians = np.median(stack, axis=0)
        mins = np.min(stack, axis=0)
        maxs = np.max(stack, axis=0)
        ranges = maxs - mins


        features = [means, stds, medians, mins, maxs, ranges]

    else:
        raise ValueError(f"Unknown feature set '{set}'. Choose 'vuong' or 'matnoise'.")

    return np.concatenate(features)
