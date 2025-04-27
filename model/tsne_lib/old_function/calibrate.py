from test_data import *
from load_data import *
from model.tsne_lib.old_function.scale_function import *
from signal_processing import *

import numpy as np
##################
calibrate_material_ref = "honey"
calibrate_material = raw

################ Calibrate Scale ################
def calibrate_raw_data(calibrate_material):
    """
    Loads and processes raw calibration material data.
    
    Returns:
        numpy.ndarray: Processed raw calibration material.
    """
    processed_data = shape_data(np.array(calibrate_material["data"]))  # Shape the raw data
    if not processed_data:
        raise ValueError("No pulses found in raw calibration data.")
    return np.array(processed_data[0])  # Convert to numpy array

def calibrate_scale(input_data, window_size):
    """
    Processes input data by trimming, smoothing, and scaling the pulses.

    Args:
        input_data (list): The raw data to be processed and scaled.

    Returns:
        list: Scaled and smoothed pulses.
    """
    # Step 1: Load Reference and Calibration Material
    ref_material = load_ref_data(calibrate_material_ref)  # Load reference data
    raw_material_calibration = calibrate_raw_data(calibrate_material)  # Load and process raw calibration data

    # ref_profiles = np.concatenate(ref_material['profile'].to_numpy())  # Flatten the array
    # ref_min = np.min(ref_profiles)
    # ref_max = np.max(ref_profiles)
    # print("Reference Min:", ref_min)
    # print("Reference Max:", ref_max)

    # Step 2: Process New Input Data
    new_sample_pulses = shape_data(input_data)  # Shape the input data
    if not new_sample_pulses:
        raise ValueError("No pulses found in input data.")
    new_sample_pulses = np.array(new_sample_pulses[0])  # Convert to numpy array

    # Step 3: Calculate and Apply Feature-Specific Scale Factors
    feature_specific_scale_factors_list = calculate_feature_specific_scale_factors(
        raw_material_calibration, ref_material
    )
    optimal_scale_factors = finalize_optimal_scale_factors(
        feature_specific_scale_factors_list, method='mean'
    )
   
    # # Call the function with the new parameters
    # scaled_pulses = apply_feature_specific_scale_factors(new_sample_pulses, optimal_scale_factors, ref_min, ref_max)

    scaled_pulses = apply_feature_specific_scale_factors(
        new_sample_pulses, optimal_scale_factors
    )

    # Step 4: Downsample the Scaled Pulses
    downsampling_data = segment_pulse(scaled_pulses.flatten())[0]  # Flatten and segment the pulse
    downsampled_pulses = downsample(downsampling_data, window_size)  # Downsample to 35 points

    # Step 5: Return the Final Scaled and Smoothed Pulses
    return downsampled_pulses
