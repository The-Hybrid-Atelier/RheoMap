# rheosense.py

import numpy as np
import pickle as pk
import pandas as pd
import os, json
from joblib import dump, load
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import pywt
from scipy.signal import medfilt
from scipy.signal import *


pca_pipe = None
lda_pipe = None
np.random.seed(0)

def load_models(DATA_DIRECTORY):
  global pca_pipe, lda_pipe
  timestamp = '20230920-020656'
  # sc0f = os.path.join(DATA_DIRECTORY, "rheosense_pca_preprocess_"+timestamp+".bin")
  # sc1f= os.path.join(DATA_DIRECTORY, "rheosense_pca_postprocess_"+timestamp+".bin")
  # pcaf= os.path.join(DATA_DIRECTORY, "rheosense_pca_"+timestamp+".pkl")
  # ldaf= os.path.join(DATA_DIRECTORY, "rheosense_lda_"+timestamp+".pkl")
  modelf= os.path.join(DATA_DIRECTORY, "rheosense_models_"+timestamp+".bin")
  # pca = pk.load(open(pcaf,'rb'))
  # sc0=load(sc0f)
  # sc1=load(sc1f)
  # pca_pipe = (sc0, pca, sc1)
  lda_pipe, pca_pipe = load(open(modelf,'rb'))
  

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



# Intializes the sensing routine
def sense(data, name, idx):
    
    pulses = segment_pulses(data)
    vp, mean, profile = pulse_profile(pulses)

    # # Debugging
    # plt.plot(np.arange(0, vp.shape[1]), profile["mean"])
    # plt.plot(np.arange(0, vp.shape[1]), profile["interval_lb"], 'r')
    # plt.plot(np.arange(0, vp.shape[1]), profile["interval_ub"], 'g')
    # plt.title(f"{vp.shape[0]} pulses detected")
    # plt.show()

    entry = {
      "id": idx,
      "name": name,
      "type": "reference",
      "pulseview": True,
      "active": True,
      "data": profile,
    }

    category, confidence = predict(mean)
    entry["data"]["lda"] = {"category": category, "confidence": confidence}
    entry["data"]["pca"] = pca_pipeline(mean)

    return entry

def predict(pulse):
    global lda_pipe
    sc0, pca, sc1 = pca_pipe
    sc0, lda, clf = lda_pipe

    features = sc0.transform([extract_features(pulse)["values"]])
    features = lda.transform(features)
    category = clf.predict(features)[0]
    confidence = max(clf.predict_proba(features)[0])

    return category, confidence

def pca_pipeline(pulse):
  global pca_pipe
  sc0, pca, sc1 = pca_pipe
  features = extract_features(pulse)["values"]
  Xp = sc0.transform([features])
  components = pca.transform(Xp)
  return sc1.transform(components)[0]


def median_filter(data, kernel_size=5):
    """Apply a median filter to the data."""
    return medfilt(data, kernel_size)

def smooth(pulse, window_size=3):
    """Smooth the pulse using a simple moving average."""
    return np.convolve(pulse, np.ones(window_size) / window_size, mode='valid')

def wavelet_filter(data, wavelet='db1', level=1):
    """Apply a Wavelet Transform based filter to denoise the data."""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(detail, np.std(detail)) for detail in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def segment_pulses(data, tolerance=1000, std_multiplier=1, target=None, pre_record=5):
    """
    Segment the data into pulses based on the given tolerance and standard deviation multiplier.
    """
    data_median_filtered = median_filter(data)
    data_wavelet_filtered = wavelet_filter(data_median_filtered)
    data = np.array(data_wavelet_filtered)
    baseline = np.bincount(data.astype(int)).argmax()

    state = 'waiting'
    pulse = []
    pulses = []
    buffer = []

    for value in data:
        deviation = abs(value - baseline)
        buffer.append(value)

        if len(buffer) > pre_record:
            buffer.pop(0)

        if state == 'waiting':
            if deviation > tolerance:
                state = 'in_pulse'
                pulse.extend(buffer)
                buffer = []
                pulse.append(value)
        elif state == 'in_pulse':
            pulse.append(value)
            if deviation <= tolerance:
                state = 'rising'
        elif state == 'rising':
            pulse.append(value)
            if deviation <= tolerance:
                state = 'waiting'
                pulses.append(pulse)
                pulse = []

    if pulse:
        pulses.append(pulse)

    # Ensure only the longest pulse is kept
    if len(pulses) > 1:
        pulses = [max(pulses, key=len)]

    lengths = [len(pulse) for pulse in pulses]
    if len(lengths) == 0:
        return []

    mean_length = np.mean(lengths)
    std_dev = np.std(lengths)
    pulses = [pulse for pulse in pulses if mean_length - std_multiplier * std_dev <= len(pulse) <= mean_length + std_multiplier * std_dev]

    if not pulses:
        return []

    max_pulse_length = max(len(pulse) for pulse in pulses) if pulses else 0
    if target:
        max_pulse_length = target

    for i in range(len(pulses)):
        difference = max_pulse_length - len(pulses[i])
        pulses[i] = pulses[i] + [baseline] * difference

    pulses = [smooth(pulse) for pulse in pulses]
    return pulses

def pulse_profile(pulses):
    if len(pulses) == 0:
        return None, None, None

    # Stack pulses
    vp = np.vstack(pulses)

    # Compute 68% interval
    mean = vp.mean(axis=0)
    sigma = vp.std(axis=0, ddof=1)

    # Handle low variation cases
    sigma[sigma == 0] = 1e-10

    # Compute the confidence intervals
    interval_lb, interval_ub = stats.norm.interval(0.68, loc=mean, scale=sigma)

    # Replace nan values with zero
    interval_lb = np.nan_to_num(interval_lb, nan=0.0)
    interval_ub = np.nan_to_num(interval_ub, nan=0.0)

    profile = {
        "mean": mean,
        "interval_lb": list(interval_lb),
        "interval_ub": list(interval_ub),
    }
    return vp, mean, profile


def extract_features(pulse):
    df = pd.DataFrame({"time": np.arange(len(pulse)), "signal": pulse})

    # Feature Extraction
    ## Shape Features
    ### Max Extrusion
    basevalue = df["signal"][0] #f1
    max_extrusion_idx = df["signal"].idxmax()
    max_extrusion_time = df["time"][max_extrusion_idx] #f6
    max_extrusion_value = df["signal"][max_extrusion_idx] #f3
    ### Min Retraction
    min_retraction_idx = df["signal"].idxmin()
    min_retraction_time = df["time"][min_retraction_idx] #f5
    min_retraction_value = df["signal"][min_retraction_idx] #f2
    ### Look only at the tail end (after max pressure value)
    tail = df.iloc[max_extrusion_idx:, :]
    ### Equilibrium
    equilibrium_idx = tail["signal"].idxmin()
    equilibrium_value = tail["signal"][equilibrium_idx] #f4
    equilibrium_time = tail["time"][equilibrium_idx] #f7
    ### Period features
    extrusion_period = max_extrusion_time - min_retraction_time #f8
    equilibrium_period = equilibrium_time - max_extrusion_time #f9
    ###  Fluid Release Point
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
    return resp

