import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pywt
from datetime import datetime
from fastdtw import fastdtw
from sklearn.metrics.pairwise import cosine_distances
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# UN-USED HEPER FUNCTION
# ============================================================================

def stack_reps(df, data_col='data', group_size=3, keep_cols=None, material=None, 
               extract_features_flag=True, feature_sets=['matnoise', 'vuong']):
    """
    Stack REPs by grouping rows and create feature vectors.
    
    Parameters:
    - df: DataFrame with pulse data
    - data_col: Name of column containing pulse arrays (default: 'data')
    - group_size: Number of rows to stack together (default: 3)
    - keep_cols: List of column names to keep from first row of each group (default: ['name'])
    - material: Optional material label to add to output (default: None)
    - extract_features_flag: Whether to extract features from stacked data (default: True)
    - feature_sets: List of feature sets to extract, options: 'matnoise', 'vuong' (default: both)
    
    Returns:
    - DataFrame with stacked data and optional features
    
    Example:
        # Basic usage - stack every 3 rows
        df_stacked = rep.stack_reps(df)
        
        # Stack with material label
        df_stacked = rep.stack_reps(df, material='plaster')
        
        # Stack without feature extraction
        df_stacked = rep.stack_reps(df, extract_features_flag=False)
        
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
    if extract_features_flag:
        if 'matnoise' in feature_sets:
            df_stacked['fluctuation_fv'] = df_stacked[data_col].apply(
                lambda x: extract_features_nocv(x, "matnoise")
            )
        
        if 'vuong' in feature_sets:
            df_stacked['geo_fv'] = df_stacked[data_col].apply(
                lambda x: extract_features_nocv(x, "vuong")
            )
    
    return df_stacked

def extract_timestamp_from_id(df):
    """Extract timestamp from MongoDB ObjectID."""
    df = df.copy()
    df['Time_Stamp'] = [
        datetime.fromtimestamp(int(str(oid)[:8], 16)) for oid in df['_id']
    ]
    return df

def iqr_outlier_filter(df, verbose=True):
    """IQR outlier filtering by 'name' group on feature vectors."""
    outlier_info = {}
    global_mask = pd.Series([False] * len(df), index=df.index)
    
    for group, subdf in df.groupby('name'):
        outlier_info[group] = {}
        if verbose:
            print(f"Material '{group}': n={len(subdf)}")
        mask = pd.Series([False] * len(subdf), index=subdf.index)
        
        for col in ['fluct_fv', 'geo_fv']:
            X = np.stack(subdf[col].values)
            centroid = X.mean(axis=0, keepdims=True)
            dists = cosine_distances(X, centroid).ravel()
            
            q1, q3 = np.percentile(dists, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = (dists < lower) | (dists > upper)
            
            mask = mask | outliers
            
            if verbose:
                print(f"  {col}: {outliers.sum()} outliers / {len(dists)}")
        
        global_mask.loc[mask.index] = global_mask.loc[mask.index] | mask
    
    df_clean = df.loc[~global_mask].copy()
    
    if verbose:
        print(f"IQR: removed {global_mask.sum()} outliers, kept {len(df_clean)}/{len(df)}")
    
    return df_clean, outlier_info
    
def generate_time_stamp(df):
    '''
    Usage:
    df = generate_time_stamp(df)
    '''
    df['Time_Stamp'] = [
        datetime.fromtimestamp(int(str(oid)[:8], 16)) for oid in df['_id']
    ]

    #calculates the difference in time between samples
    df['Time_Elapsed (s)'] = (df['Time_Stamp'] - df['Time_Stamp'].min()).dt.total_seconds()

    # Relative time within each name (in seconds)
    df['Relative_time_elapsed (s)'] = (
        df.groupby('name')['Time_Stamp']
            .transform(lambda s: (s - s.min()).dt.total_seconds())
    )
    df = df.sort_values(by=["name", "Relative_time_elapsed (s)"]).reset_index(drop=True)
    return df
