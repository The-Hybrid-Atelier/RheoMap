import numpy as np
import pandas as pd
import dill
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures

PATH = "/workspaces/RheoPulse/model/master_dataset/transformer.dill"
length = 35

def resample_to_match_length(data, target_length):
    """Interpolates or resamples data to match the target length."""
    x_original = np.linspace(0, 1, len(data))
    x_target = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_original, data, kind='linear')
    return interpolator(x_target)

def transform_fn(pulse: np.ndarray,
                        params: np.ndarray,
                        ref_length: int) -> np.ndarray:
    """
    Transform a single timeseries using the given parameters.
    """
    mean_phase_shift = int(params[0])
    intercept = params[1]
    coefficients = params[2:]

    # Resample and shift
    shifted_pulse = np.roll(resample_to_match_length(pulse, ref_length),
                           mean_phase_shift)

    # Apply polynomial transformation
    poly = PolynomialFeatures(degree=len(coefficients)-1)
    X = poly.fit_transform(shifted_pulse.reshape(-1, 1))
    transformed = X @ coefficients + intercept

    return transformed

def apply_transform_and_extract_features(input_data):
    """
    Loads a transformer from a dill file, applies transformation, and extracts features.

    Args:
        input_data (pd.DataFrame): DataFrame with a column named 'pulse', containing the pulse data to process.
        dill_path (str): Path to the dill file containing the transformer.
        ref_length (int): Reference length for the transformer (default is 40).

    Returns:
        pd.DataFrame: Updated DataFrame with 'transformed_pulse' and 'scaled_features' columns.
    """
    dill_path = PATH
    ref_length = length
    
    with open(dill_path, "rb") as file:
        loaded_transformer = dill.load(file)
    # print("Transformer loaded successfully:", type(loaded_transformer))

    # Verify 'pulse' column exists
    if 'pulse' not in input_data.columns:
        raise ValueError("Input DataFrame must contain a 'pulse' column.")
    # Apply transformation and extract features for the input dataset
    try:
        input_data['transformed_pulse'] = input_data['pulse'].apply(lambda pulse: loaded_transformer.transform(pulse, ref_length))
        input_data['scaled_features'] = input_data['transformed_pulse'].apply(lambda transformed_pulse: loaded_transformer.extract_signal_features(transformed_pulse))
    except Exception as e:
        print(f"Error during transformation or feature extraction: {e}")
        raise

    # Return the updated DataFrame
    return input_data

def extract_features(pulse, new_data=False):
    df = pd.DataFrame({"time": np.arange(len(pulse)), "signal": pulse})
    # Feature Extraction
    ## Shape Features
    ### Max Extrusion
    # basevalue = df["signal"][0] #f1
    basevalue = np.mean(df["signal"][:3]) #f1
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
