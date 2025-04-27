from load_data import *
from feature_extract import *

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# ############## Load Global Data for Processing ##############
submap, scaler = load_submap_data()  # Load submap data and scaler globally
master_data = load_masterdata()  # Load master data
tsne_coordinates = submap[['TSNE1', 'TSNE2']].to_numpy()
submap_materials = submap['material'].unique()

# Filter master data to include only relevant materials
master_data = master_data[master_data['material'].isin(submap_materials)].reset_index(drop=True)

# Features

FEATURES = ['basevalue',
 'min_retraction_value',
 'max_extrusion_value',
 'equilibrium_value',
 'max_retraction_time',
 'max_extrusion_time',
 'equilibrium_time',
 'extrusion_period',
 'equilibrium_period',
 'fluid_release_point_time',
 'fluid_release_point_value',
 'fluid_release_point_period']

# ############## Project to t-SNE Space ##############
def project_to_tsne_space(new_data_point, k, scaler=None):
    """
    Projects a new data point onto the t-SNE space using k-nearest neighbors and inverse distance weighting.

    Args:
        new_data_point (list or dict): The data point to project.
        k (int): Number of neighbors to consider for projection.
        scaler (StandardScaler, optional): Scaler object for feature scaling.

    Returns:
        np.ndarray: The projected t-SNE coordinates.
    """
    
    # Step 1: Expand profile in master data if necessary
    if 'profile' in master_data.columns:
        profile_df = pd.DataFrame(master_data['profile'].to_list()).add_prefix('profile_')
        original_data_expanded = pd.concat([master_data.drop(columns='profile'), profile_df], axis=1)
    else:
        original_data_expanded = master_data

    # Step 2: Extract and scale features
    original_features = original_data_expanded[FEATURES].fillna(0).values
    if len(tsne_coordinates) != len(original_features):
        raise ValueError(f"Mismatch in data lengths: tsne_coordinates has {len(tsne_coordinates)}, original_features has {len(original_features)}")

    if scaler is None:
        scaler = StandardScaler()
        original_features_scaled = scaler.fit_transform(original_features)
    else:
        original_features_scaled = scaler.transform(original_features)

    # Prepare new data point for projection
    if isinstance(new_data_point, list):
        new_data_point = pd.DataFrame([new_data_point], columns=FEATURES)
    elif isinstance(new_data_point, dict):
        new_data_point = pd.DataFrame([new_data_point])

    new_features_scaled = scaler.transform(new_data_point[FEATURES].fillna(0).values)

    # Step 3: Set up k-NN with Mahalanobis distance
    covariance_matrix = np.cov(original_features_scaled, rowvar=False)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='mahalanobis', metric_params={'VI': np.linalg.inv(covariance_matrix)})
    nbrs.fit(original_features_scaled)
    distances, indices = nbrs.kneighbors(new_features_scaled)

    # Step 4: Perform inverse distance weighting
    weights = 1 / (distances[0] + 1e-8)
    weights /= weights.sum()  # Normalize weights
    neighbors_tsne = tsne_coordinates[indices[0]]
    new_tsne_point = np.average(neighbors_tsne, axis=0, weights=weights)

    return new_tsne_point