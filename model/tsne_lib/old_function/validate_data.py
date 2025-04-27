""" From 5 different material classes: [milk, ketchup, mayo, honey, canola_oil]
    Extract 50 random pulses evenly from the material classes  from the PA dataset. - X
    Extract 50 random pulses evenly from the material classes from the ThingPlus dataset.
    Extract 10 different reference water pulses from the PA dataset. - X
    Extract 10 reference water pulses from the ThingPlus dataset - X
    This should be a binary called validation_data.bin and loaded into a new notebook. - X
    For each material class, apply your scaling function and plot all 20 pulses on a graph. There should be 10 PA and 10 TP.
    Then, generate the features for all 20 pulses and plot the boxplot per feature. We should have 9(?) features and 18 boxplots in a graph. 2 boxplots (PA/TP) per feature.
 """
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from load_data import *
from feature_extract import *
from plot_data import *
from test_data import *
from scipy.interpolate import interp1d

materials_to_calibrate = ["milk", "ketchup", "mayo", "honey", "canola_oil"]
validation_path = '/workspaces/RheoPulse/model/master_dataset/validation_data.bin'
num_pulses_per_class=10
num_reference_pulses=10


def scale_test(array, rangeA=[0, 160000], rangeB=[0, 900], kind='linear'):
    minA, maxA = rangeA
    minB, maxB = rangeB

    # Create an interpolation function
    interpolator = interp1d([minA, maxA], [minB, maxB], kind=kind, fill_value="extrapolate")

    # Apply the interpolation to the array
    arr = interpolator(array)

    # Center the data around the midpoint of rangeB
    midpointB = (maxB + minB) / 2
    scaled_pulse = arr - np.mean(arr) + midpointB

    # Flatten and segment the pulse
    scaled_pulse = scaled_pulse.flatten()
    return scaled_pulse

def extract_random_pulses(data, materials, num_pulses_per_class, datatype_label):
    """
    Extracts an equal number of random pulses from each specified material class
    and adds a 'datatype' column to indicate the data source.
    
    Parameters:
    - data (DataFrame): The input data containing multiple material classes.
    - materials (list): A list of materials to filter and sample from.
    - num_pulses_per_class (int): The number of pulses to sample from each material class.
    - datatype_label (str): A label to indicate the data source (e.g., "pa" or "tp").
    
    Returns:
    - DataFrame: A new DataFrame containing the sampled pulses with a 'datatype' column.
    """
    # Initialize an empty DataFrame to store the sampled pulses
    sampled_data = pd.DataFrame()

    # Extract specified number of random pulses from each material class
    for material in materials:
        material_data = data[data['material'] == material]
        if len(material_data) >= num_pulses_per_class:
            sampled_material_data = material_data.sample(n=num_pulses_per_class, random_state=42)
            sampled_material_data = sampled_material_data.copy()  # To avoid SettingWithCopyWarning
            sampled_material_data['datatype'] = datatype_label  # Add the datatype label
            sampled_data = pd.concat([sampled_data, sampled_material_data], ignore_index=True)
        else:
            print(f"Warning: Not enough data for material '{material}' to sample {num_pulses_per_class} pulses.")

    return sampled_data

def save_data(data, filename):
    # Save data as binary
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

def load_validation_data(filename):
    # Load binary data
    with open(filename, 'rb') as file:
        return pickle.load(file)

def create_validation_data():
    """
    Loads PA and TP datasets, extracts samples for calibration and reference,
    adds a 'datatype' column, combines data, and saves it as binary.

    Parameters:
    - materials_to_calibrate (list): List of materials to sample for calibration (e.g., ["milk", "ketchup", "mayo", "honey", "canola_oil"]).
    - validation_path (str): Path to save the combined validation data.
    - num_pulses_per_class (int): Number of random pulses to sample per material class.
    - num_reference_pulses (int): Number of reference water pulses to sample.

    Returns:
    - DataFrame: Combined validation data with 'datatype' column.
    """

    # Load PA and ThingPlus datasets from files
    pa_data = load_masterdata()
    tp_data = load_masterdata_thingplus() 
    # tp_data = load_masterdata() ######################################################### ThingPlus data

    # Extract 50 random pulses for each material class from PA and TP datasets
    pa_data_sample = extract_random_pulses(pa_data, materials_to_calibrate, num_pulses_per_class, datatype_label="pa")
    tp_data_sample = extract_random_pulses(tp_data, materials_to_calibrate, num_pulses_per_class, datatype_label="tp")

    # Extract 10 reference water pulses for each dataset and add 'datatype' column
    pa_ref_data_sample = pa_data[pa_data['material'] == 'water'].sample(n=num_reference_pulses, random_state=42).copy()
    pa_ref_data_sample['datatype'] = "pa"
    
    tp_ref_data_sample = tp_data[tp_data['material'] == 'water'].sample(n=num_reference_pulses, random_state=42).copy()
    tp_ref_data_sample['datatype'] = "tp"

    # Combine all samples into validation data
    validation_data = pd.concat([pa_data_sample, tp_data_sample, pa_ref_data_sample, tp_ref_data_sample], ignore_index=True)

    # Save validation data to binary file
    with open(validation_path, 'wb') as file:
        pickle.dump(validation_data, file)
    print(f"Data saved to {validation_path}")

    # Return the combined validation data
    validation_data = validation_data[['material', 'profile', 'basevalue',
                                         'min_retraction_value', 'max_extrusion_value', 'equilibrium_value',
                                         'max_retraction_time', 'max_extrusion_time', 'equilibrium_time',
                                         'extrusion_period', 'equilibrium_period', 'fluid_release_point_time',
                                         'fluid_release_point_value', 'fluid_release_point_period', 'datatype']]
    return validation_data

#################### Boxplot Calibration ####################
def boxplot_calibration(validation_data):
    """
    Processes and plots pulses for each material class in the validation data,
    extracts features for each pulse, and generates boxplots comparing PA and TP.

    Parameters:
    - validation_data (DataFrame): The combined data from `create_validation_data`.
    """

    # Initialize a dictionary to store features for each material and datatype
    features_dict = {material: {'pa': [], 'tp': []} for material in materials_to_calibrate}

    # Process and plot the pulses for each material class
    for material in materials_to_calibrate:
        # Separate PA and TP pulses for the material
        pa_pulses = validation_data[(validation_data['material'] == material) & (validation_data['datatype'] == 'pa')]
        tp_pulses = validation_data[(validation_data['material'] == material) & (validation_data['datatype'] == 'tp')]

        # Use unscaled pulses directly for PA data
        unscaled_pa_pulses = list(pa_pulses['profile'])
        
        # Apply scaling to TP pulses only
        scaled_tp_pulses = [scale_test(pulse) for pulse in tp_pulses['profile']]
        # scaled_tp_pulses = list(tp_pulses['profile'])
        # Plot PA pulses (unscaled)
        plot_pulse_calibrate_data(unscaled_pa_pulses, title=f"{material} - Pulses (PA)")
        
        # Plot TP pulses (scaled)
        plot_pulse_calibrate_data(scaled_tp_pulses, title=f"{material} - Scaled Pulses (TP)")

        # Store existing PA features directly from validation_data
        for _, row in pa_pulses.iterrows():
            features_dict[material]['pa'].append({feature: row[feature] for feature in FEATURES})
        
        # Extract features for TP pulses and store in features_dict
        for pulse in scaled_tp_pulses:
            extracted_features = extract_features(pulse)
            features_dict[material]['tp'].append(dict(zip(extracted_features["names"], extracted_features["values"])))
    
    # Generate boxplots for each feature, with PA and TP side by side
    for feature_name in FEATURES:
        boxplot_data = {'PA': [], 'TP': []}
        
        for material in materials_to_calibrate:
            # Extract feature values from the feature dictionaries, handling missing features gracefully
            pa_features = [f.get(feature_name, None) for f in features_dict[material]['pa']]
            tp_features = [f.get(feature_name, None) for f in features_dict[material]['tp']]
            
            # Filter out any None values to avoid plotting errors
            pa_features = [value for value in pa_features if value is not None]
            tp_features = [value for value in tp_features if value is not None]
            
            # Add the extracted feature data to boxplot_data
            boxplot_data['PA'].extend(pa_features)
            boxplot_data['TP'].extend(tp_features)
        
        # Plot boxplot for the current feature
        plot_boxplot(boxplot_data, title=f"Boxplot of {feature_name} (PA vs TP)")

# # Load the validation data and plot the calibration results
# validata_data_test = create_validation_data()
# print(validata_data_test)
# boxplot_calibration(validata_data_test)
