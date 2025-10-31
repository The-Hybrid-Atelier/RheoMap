import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pywt

# REP (Retraction-Extrusion Pulse 2 Sensor Vector) 
# Methods that can be applied to a dataframe to obtain:
# 1. Plotting of pulses by condition
# 2. Filtering of pulses (median, wavelet, moving average)
# 3. Feature extraction from pulses

# Dataframe should have a column with pulses (e.g., must be equal length). 
# For visualization, dataframe should have a condition column (e.g., 'material', etc.).


def plot_lines_by_condition(df, pulse_col, condition_col):
    """
    Plot all padded pulses as lines, grouped by condition.
    """
    pulse_matrix = np.vstack(df[pulse_col].to_numpy())
    n_trials, pulse_length = pulse_matrix.shape

    plot_df = pd.DataFrame({
        'value': pulse_matrix.flatten() / 1000,
        'time_index': np.tile(np.arange(pulse_length), n_trials),
        'condition': np.repeat(df[condition_col].values, pulse_length)
    })

    sns.set(style="whitegrid", context="talk")
    unique_conditions = plot_df["condition"].unique()
    palette = sns.color_palette("Set2", len(unique_conditions))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x='time_index',
        y='value',
        hue='condition',
        linewidth=3,
        errorbar='sd',
        palette=palette,
        ax=ax
    )

    ax.set_xlabel("Sample", fontsize=16)
    ax.set_ylabel("Air Pressure (kPa)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(title="Condition", fontsize=14, title_fontsize=16)

    fig.tight_layout()
    return fig, ax 


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
    pulse_med = medfilt(pulse, kernel_size=median_k)

    # Step 2: Wavelet denoising
    coeffs = pywt.wavedec(pulse_med, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(detail, np.std(detail)) for detail in coeffs[1:]]
    pulse_wave = pywt.waverec(coeffs, wavelet)

    # Match original length in case waverec adds samples
    pulse_wave = pulse_wave[:len(pulse_med)]

    # Step 3: Moving average
    pulse_smooth = np.convolve(pulse_wave, np.ones(window_size) / window_size, mode='valid')

    return pulse_smooth

def extract_features(pulse):
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


def stack_reps(df, data_col='data', group_size=3, keep_cols=None, material=None, 
               extract_features=True, feature_sets=['matnoise', 'vuong']):
    """
    Stack REPs by grouping rows and create feature vectors.
    
    Parameters:
    - df: DataFrame with pulse data
    - data_col: Name of column containing pulse arrays (default: 'data')
    - group_size: Number of rows to stack together (default: 3)
    - keep_cols: List of column names to keep from first row of each group (default: ['name'])
    - material: Optional material label to add to output (default: None)
    - extract_features: Whether to extract features from stacked data (default: True)
    - feature_sets: List of feature sets to extract, options: 'matnoise', 'vuong' (default: both)
    
    Returns:
    - DataFrame with stacked data and optional features
    
    Example:
        # Basic usage - stack every 3 rows
        df_stacked = rep.stack_reps(df)
        
        # Stack with material label
        df_stacked = rep.stack_reps(df, material='plaster')
        
        # Stack without feature extraction
        df_stacked = rep.stack_reps(df, extract_features=False)
        
        # Stack every 5 rows with only 'vuong' features
        df_stacked = rep.stack_reps(df, group_size=5, feature_sets=['vuong'])
        
        # Keep additional columns and specify material later
        df_stacked = rep.stack_reps(df, keep_cols=['name', 'trial_id'])
        df_stacked['material'] = 'plaster'  # Add material later
    """
    if keep_cols is None:
        keep_cols = ['name']
    
    grouped_data = []
    
    # Group by specified group_size and stack the data column
    for i in range(0, len(df), group_size):
        group = df.iloc[i:i+group_size]
        
        if len(group) == group_size:  # Ensure we have full groups
            # Stack the data arrays
            stacked_data = np.stack(group[data_col].values)
            
            # Create result dictionary with stacked data
            result = {data_col: stacked_data}
            
            # Keep specified columns from the first row of the group
            for col in keep_cols:
                if col in group.columns:
                    result[col] = group.iloc[0][col]
            
            grouped_data.append(result)
    
    # Create new dataframe with stacked data
    df_stacked = pd.DataFrame(grouped_data)
    
    # Add material column if specified
    if material is not None:
        df_stacked['material'] = material
    
    # Extract features if requested
    if extract_features:
        if 'matnoise' in feature_sets:
            df_stacked['fluctuation_fv'] = df_stacked[data_col].apply(
                lambda x: extract_features_nocv(x, "matnoise")
            )
        
        if 'vuong' in feature_sets:
            df_stacked['geo_fv'] = df_stacked[data_col].apply(
                lambda x: extract_features_nocv(x, "vuong")
            )
    
    return df_stacked
