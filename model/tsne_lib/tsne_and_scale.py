from projection import *
from plot_data import *
from test_data import *
from feature_extract import *
from scipy.signal import medfilt
import pywt
import numpy as np

######## Initialize the data ########
k_value = 3 # Number of neighbors for t-SNE projection
test_raw_data = clay2 # Material from ThingPlus
material_name = "honey"  # Material from Masterdata PA

######## Define the features ########
def smooth(pulse, window_size=3):
  return np.convolve(pulse, np.ones(window_size)/window_size, mode='valid')

def median_filter(data, kernel_size=5):
    """
    Apply a median filter to the data.
    - kernel_size: size of the kernel, must be a positive odd integer
    """
    return medfilt(data, kernel_size)

def wavelet_filter(data, wavelet='db1', level=1):
    """
    Apply a Wavelet Transform based filter to denoise the data.
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs[1:] = (pywt.threshold(detail, np.std(detail)) for detail in coeffs[1:])
    return pywt.waverec(coeffs, wavelet)

def segment_pulses_thingplus(data, tolerance=1000, std_multiplier=1, target=None, pre_record=5):

    # Noise filtering using median filter
    data_median_filtered = median_filter(data)
    data_wavelet_filtered = wavelet_filter(data_median_filtered)

    # Data Stats:
    data = np.array(data_wavelet_filtered)  # Ensure data is a numpy array
    mean = np.mean(data)
    std_dev = np.std(data)
    baseline = np.bincount(data.astype(int)).argmax()  # Most common value as baseline

    # Initial State:
    state = 'waiting'  # can be 'waiting', 'in_pulse', 'rising'
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
                pulse.extend(buffer)  # Add buffered values
                buffer = []  # Clear the buffer after transferring
                pulse.append(value)
            else:
                # Not yet in pulse, keep filling the buffer
                pass

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

    # In case there's a pulse at the end of the data which didn't return to the baseline
    if pulse:
        pulses.append(pulse)

    # Remove outlier pulses based on length
    lengths = [len(pulse) for pulse in pulses]
    mean_length = np.mean(lengths)
    std_dev = np.std(lengths)
    pulses = [pulse for pulse in pulses if mean_length - std_multiplier * std_dev <= len(pulse) <= mean_length + std_multiplier * std_dev]

    # Determine the size of the largest pulse
    max_pulse_length = max(len(pulse) for pulse in pulses) if pulses else 0

    if target:
        max_pulse_length = target

    # Pad pulses to ensure they're all the same size
    for i in range(len(pulses)):
        difference = max_pulse_length - len(pulses[i])
        pulses[i] = pulses[i] + [baseline] * difference

    # Smooth the pulses
    pulses = [smooth(pulse) for pulse in pulses]

    return pulses

def tnse_and_scale(input_data):
    """
    Process raw pulse data, segment and scale the pulses, extract features, and project the data into t-SNE space.

    Args:
        input_data (dict): Dictionary containing raw pulse data.
        k_value (int): Number of nearest neighbors for t-SNE projection.
        scaler (StandardScaler, optional): Scaler object for feature scaling.

    Returns:
        tuple: New t-SNE coordinates (numpy array) for the projected data point, scaled pulses, and transformed data.
    """
    # Step 1: Access and prepare pulse data from the 'data' field directly
    data = pd.DataFrame()
    data['method'] = ["TP"]
    data['material'] = [input_data['params']['material']]
    data['pulse'] = [input_data['data']]
    data['pulse'] = [segment_pulses_thingplus(input_data['data'])[0]]
    data['sensor_vector'] = [[]]
    transformed_data = apply_transform_and_extract_features(data)
    # Step 4: Prepare data point for t-SNE projection
    feature_data = {feature: transformed_data[feature].iloc[0] for feature in transformed_data.columns if feature != 'pulse'}

    features = extract_features(feature_data['transformed_pulse'])['values']
    final_features = pd.DataFrame([features],columns=FEATURES)
    # Step 5: Project the extracted features into t-SNE space
    new_tsne_point = project_to_tsne_space(
        new_data_point=final_features, 
        k=k_value,
        scaler=scaler
    )
    return new_tsne_point, transformed_data['transformed_pulse'].values, transformed_data

def process_and_plot(raw_data):
    """
    Processes raw pulse data, scales it, projects it into t-SNE space, and plots the results.

    Args:
        raw_data (dict): Raw pulse data from ThingPlus.
        material_name (str): Material name from Masterdata PA.
        k_value (int, optional): Number of neighbors for t-SNE projection. Default is 5.
    """
    # Perform t-SNE projection and scale the pulses
    new_tsne_coordinates, scaled_pulses, transformed_data = tnse_and_scale(raw_data)
    # print("New t-SNE coordinates for the projected data point:", new_tsne_coordinates)
    
    # Plot the original pulse data, scaled pulse, master data, and t-SNE projection
    plot_orignal_pulse_data(raw_data)
    plot_scaled_pulse(scaled_pulses)
    plot_master_data(material_name)
    # box_plot_feature(transformed_data, material_name)
    # plot_tsne(new_tsne_coordinates, raw_data)
    # plot_tsne_with_ref(new_tsne_coordinates, raw_data, material_name)
    # plot_chi(new_tsne_coordinates, raw_data)
    # plot_chi2(new_tsne_coordinates, raw_data, material_name)
    return new_tsne_coordinates

process_and_plot(test_raw_data)
