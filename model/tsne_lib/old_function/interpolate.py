from signal_processing import *
from test_data import *
from feature_extract import *
from load_data import *
from model.tsne_lib.old_function.validate_data import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

def interpolate_array(array, rangeA=[0, 160000], rangeB=[0, 900], kind='linear'):
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
    segmented_pulse = segment_pulse(scaled_pulse)
    return segmented_pulse[0] if segmented_pulse else scaled_pulse

def interpolate_up(array, rangeA=[0, 900], rangeB=[0, 160000], kind='linear'):
    minA, maxA = rangeA
    minB, maxB = rangeB

    # Create an interpolation function from rangeA to rangeB
    interpolator = interp1d([minA, maxA], [minB, maxB], kind=kind, fill_value="extrapolate")

    # Apply the interpolation to the array
    arr = interpolator(array)

    # Center the data around the midpoint of rangeB
    midpointB = (maxB + minB) / 2
    scaled_pulse = arr - np.mean(arr) + midpointB

    # Flatten and segment the pulse
    scaled_pulse = scaled_pulse.flatten()
    return scaled_pulse


############ Generate ThingPlus Data ############
# tp_data = load_masterdata()
# materials_to_calibrate = ["water","milk", "ketchup", "mayo", "honey", "canola_oil"]
# num_pulses_per_class = 20
# tp_data_sample = extract_random_pulses(tp_data, materials_to_calibrate, num_pulses_per_class, datatype_label="tp")
# tp_data_sample['profile'] = tp_data_sample['profile'].apply(lambda x: interpolate_up(np.array(x)))
# tp_data_sample = tp_data_sample.drop(columns=['datatype'])
# filename = '/workspaces/RheoPulse/model/master_dataset/thingplus.bin'

# # Save the DataFrame to a binary file
# with open(filename, 'wb') as file:
#     pickle.dump(tp_data_sample, file)
# print(f"Data saved to {filename}")
# charlie = load_masterdata_thingplus()
# print(charlie)