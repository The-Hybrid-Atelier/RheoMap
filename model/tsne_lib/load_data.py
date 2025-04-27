import pandas as pd
import os
from joblib import load  
import glob
import pickle

master_farme_path = '/workspaces/RheoPulse/model/master_dataset/master_frame_clean_*.bin'
# master_farme_path = '/Users/atelier-member/Documents/GitHub/RheoPulse/model/master_data/master_frame_clean_*.bin'
tsne_path = '/workspaces/RheoPulse/model/master_dataset/submap_*.bin'
# tsne_path = '/Users/atelier-member/Documents/GitHub/RheoPulse/model/master_data/submap_*.bin'

############ Load Master Data ############
def load_masterdata(file_pattern=master_farme_path):
    """
    Loads the most recent t-SNE results from the current local directory, extracts the submap and scaler data,
    and converts the submap into a DataFrame with appropriate columns.

    Args:
        file_pattern (str): Pattern to match the submap files (default is 'submap_*.bin').

    Returns:
        pd.DataFrame: DataFrame containing the loaded submap data.
        dict: The loaded scaler object.
    """
    # 1. Set the current directory path (no need to change directory)
    current_directory = os.getcwd()

    # 2. Find the latest file matching the pattern in the current directory
    list_of_files = glob.glob(os.path.join(current_directory, file_pattern))
    if not list_of_files:
        raise FileNotFoundError(f"No '{file_pattern}' files found in the current directory.")

    latest_file = max(list_of_files, key=os.path.getctime)  # Get the latest file

    # 3. Load the most recent t-SNE results
    loaded_data = load(latest_file)  # Load the binary data

    return loaded_data

############## Load T-SNE Map ##############
def load_submap_data(file_pattern=tsne_path):
    """
    Loads the most recent t-SNE results from the current local directory, extracts the submap and scaler data,
    and converts the submap into a DataFrame with appropriate columns.

    Args:
        file_pattern (str): Pattern to match the submap files (default is 'submap_*.bin').

    Returns:
        pd.DataFrame: DataFrame containing the loaded submap data.
        dict: The loaded scaler object.
    """
    # 1. Set the current directory path (no need to change directory)
    current_directory = os.getcwd()

    # 2. Find the latest file matching the pattern in the current directory
    list_of_files = glob.glob(os.path.join(current_directory, file_pattern))
    if not list_of_files:
        raise FileNotFoundError(f"No '{file_pattern}' files found in the current directory.")

    latest_file = max(list_of_files, key=os.path.getctime)  # Get the latest file

    # 3. Load the most recent t-SNE results
    loaded_data = load(latest_file)  # Load the binary data
    if 'submap' not in loaded_data or 'scaler' not in loaded_data:
        raise ValueError(f"'submap' or 'scaler' not found in the loaded data from {latest_file}.")

    # 4. Extract the loaded data
    submap_loaded = loaded_data['submap']
    scaler_loaded = loaded_data['scaler']

    # 5. Define submap columns
    submap_columns = ['TSNE1', 'TSNE2', 'material', 'trial', 'profile', 'version',
                      'basevalue', 'min_retraction_value', 'max_extrusion_value',
                      'equilibrium_value', 'max_retraction_time', 'max_extrusion_time',
                      'equilibrium_time', 'extrusion_period', 'equilibrium_period',
                      'fluid_release_point_time', 'fluid_release_point_value',
                      'fluid_release_point_period', 'experiment', 'type', 'pca',
                      'feature', 'value', 'max_extrusion_valueloss',
                      'min_retraction_valueloss', 'total_loss', 'viscosity', 'mattype']

    # 6. Convert submap array into a DataFrame
    submap_df = pd.DataFrame(submap_loaded, columns=submap_columns)

    return submap_df, scaler_loaded
