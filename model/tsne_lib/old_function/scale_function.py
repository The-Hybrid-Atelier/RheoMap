from feature_extract import *
import numpy as np
import pandas as pd

############ Scale Pulse Data ############
def calculate_feature_specific_scale_factors(raw_material_calibration, ref_material_df):
    """
    Calculate feature-specific scale factors for a raw pulse by comparing it to each reference pulse.

    Args:
        raw_material_calibration (array): A single pulse of raw data (e.g., raw_water).
        ref_material_df (DataFrame): DataFrame containing the reference pulses with feature columns.

    Returns:
        list: A list of dictionaries where each dictionary contains feature-specific scale factors 
              for each reference pulse.
    """
    # Extract features from the raw material calibration
    raw_features = extract_features(raw_material_calibration)
    raw_feature_names = raw_features['names']
    raw_feature_values = raw_features['values']

    # Initialize a list to store the feature-specific scale factors
    feature_specific_scale_factors_list = []

    # Iterate over each row in the reference material DataFrame
    for _, row in ref_material_df.iterrows():
        # Extract the reference pulse data and its features
        reference_pulse = row['profile']  # 'profile' should contain the pulse data for the material reference
        reference_features = extract_features(reference_pulse)['values']

        # Compute scale factors for each feature
        scale_factors = {
            feature_name: calculate_scale_factor(raw_value, ref_value)
            for feature_name, raw_value, ref_value in zip(raw_feature_names, raw_feature_values, reference_features)
        }

        # Append the scale factors for this reference to the list
        feature_specific_scale_factors_list.append(scale_factors)

    return feature_specific_scale_factors_list

def calculate_scale_factor(raw_value, ref_value):
    """
    Calculate the scale factor between a raw value and a reference value.

    Args:
        raw_value (float, array, or list): The raw feature value(s).
        ref_value (float, array, or list): The reference feature value(s).

    Returns:
        float: The calculated scale factor. Returns 0 if the raw value is 0 to avoid division by zero.
    """
    if isinstance(raw_value, (np.ndarray, list)):
        # For arrays or lists, use the mean value for comparison
        raw_value_mean = np.mean(raw_value)
        ref_value_mean = np.mean(ref_value) if isinstance(ref_value, (np.ndarray, list)) else ref_value
        return ref_value_mean / raw_value_mean if raw_value_mean != 0 else 0
    else:
        # For scalar values, compute the scale factor directly
        return ref_value / raw_value if raw_value != 0 else 0

def finalize_optimal_scale_factors(scale_factors_list, method='mean'):
    """
    Finalize optimal scale factors for each feature based on multiple reference pulses.
    
    Args:
        scale_factors_list (list): List of dictionaries containing feature-specific scale factors across reference pulses.
        method (str): Method to summarize scale factors ('mean', 'median', etc.).
    
    Returns:
        pd.Series: Series with the final optimal scale factor for each feature.
    """
    # Convert the list of dictionaries into a DataFrame for easy processing
    feature_scale_factors_df = pd.DataFrame(scale_factors_list)

    if method == 'mean':
        optimal_scale_factors = feature_scale_factors_df.mean()
    elif method == 'median':
        optimal_scale_factors = feature_scale_factors_df.median()
    else:
        raise ValueError(f"Unknown method {method}. Supported methods: 'mean', 'median'")

    # print(f"Final optimal scale factors (using {method}):\n{optimal_scale_factors}")
    return optimal_scale_factors

def apply_feature_specific_scale_factors(new_pulse, scale_factors):
    """
    Scale the entire pulse by `basevalue` and adjust specific sections based on 
    `min_retraction_value`, `max_extrusion_value`, and `equilibrium_value`.
    
    Args:
        new_pulse (array): The original pulse data to be scaled.
        scale_factors (pd.Series): Series containing the optimal scale factors for each feature.
    
    Returns:
        array: The scaled pulse data.
    """
    # Convert pulse to float to avoid integer casting issues
    new_pulse_scaled = np.array(new_pulse, dtype=np.float64)
    
    # Extract features from the pulse
    features = extract_features(new_pulse)
    
    # Base scaling with basevalue
    basevalue_factor = scale_factors.get("basevalue", 1)
    new_pulse_scaled *= basevalue_factor
    
    # Adjust segments based on min_retraction, max_extrusion, and equilibrium
    min_retraction_index = int(features["values"][4])
    max_extrusion_index = int(features["values"][5])
    equilibrium_index = int(features["values"][6])

    # Min retraction adjustment - subtracts scaled value up to min_retraction_index
    min_retraction_factor = scale_factors.get("min_retraction_value", 1)
    new_pulse_scaled[:min_retraction_index] -= min_retraction_factor 

    # Max extrusion adjustment - adds scaled value from max_extrusion_index onward
    max_extrusion_factor = scale_factors.get("max_extrusion_value", 1)
    new_pulse_scaled[max_extrusion_index:] += max_extrusion_factor 

    return new_pulse_scaled
