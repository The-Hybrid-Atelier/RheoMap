import numpy as np
from scipy.signal import medfilt
import pywt

############ Signal Processing Functions ############
def smooth(pulse, window_size=3):
    """Smooth the pulse using a simple moving average."""
    return np.convolve(pulse, np.ones(window_size) / window_size, mode='valid')

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

def remove_trailing_baseline(pulse, baseline, range_threshold=10):
    """
    Remove trailing values close to the baseline at the end of the pulse.
    
    Parameters:
    - pulse: The pulse data array
    - baseline: The baseline value to compare against
    - range_threshold: The maximum deviation from the baseline to consider a value as part of the tail
    
    Returns:
    - Trimmed pulse data
    """
    # Traverse from the end of the pulse and remove values close to the baseline
    for i in range(len(pulse) - 1, -1, -1):
        if abs(pulse[i] - baseline) > range_threshold:
            # Keep all values up to and including this point
            return pulse[:i + 1]
    return pulse  # If all values are within the range, return the whole pulse

def segment_pulse(data, tolerance=3, std_multiplier=1, target=None, pre_record=5):

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
    # pulses = [smooth(pulse) for pulse in pulses]
    pulses = [smooth(remove_trailing_baseline(pulse, baseline)) for pulse in pulses]

    return pulses

def shape_data(data, target=None):
    """
    Format the data to preserve its original length without segmentation 
    or padding, while preserving debug information.
    """
    # Flatten the data and check shape
    data = np.array(data).flatten()  # Ensure it's a 1D array
    # print(f"Data shape: {data.shape}")

    # Compute baseline using the most frequent value (mode)
    baseline = np.bincount(data.astype(int)).argmax()
    # print(f"Baseline: {baseline}")

    # Mock pulse detection for debug purposes
    pulses = [data]  # Assume all data is one pulse in this case
    # print(f"Number of pulses detected: {len(pulses)}")
    
    # Keep only the longest pulse (the full data in this case)
    longest_pulse = max(pulses, key=len)
    # print(f"Longest pulse length: {len(longest_pulse)}")
    
    # Return the original pulse without padding
    return [longest_pulse]


def downsample(data, target_length, method='mean'):
    """
    Downsamples the input data to the specified target length.

    Parameters:
    - data (array-like): The original data to be downsampled.
    - target_length (int): The desired length of the downsampled data.
    - method (str): The method to use for downsampling, one of ['mean', 'median', 'max', 'min'].

    Returns:
    - np.array: The downsampled data.
    """
    if target_length <= 0:
        raise ValueError("Target length must be a positive integer.")
    if method not in ['mean', 'median', 'max', 'min']:
        raise ValueError("Method must be one of ['mean', 'median', 'max', 'min'].")

    # Calculate the downsampling factor
    factor = len(data) / target_length
    
    # Choose downsampling method
    downsampled_data = []
    for i in range(target_length):
        start = int(i * factor)
        end = int((i + 1) * factor)
        
        # Ensure we handle the last segment correctly
        if i == target_length - 1:
            segment = data[start:]
        else:
            segment = data[start:end]
        
        if method == 'mean':
            downsampled_data.append(np.mean(segment))
        elif method == 'median':
            downsampled_data.append(np.median(segment))
        elif method == 'max':
            downsampled_data.append(np.max(segment))
        elif method == 'min':
            downsampled_data.append(np.min(segment))

    return np.array(downsampled_data)