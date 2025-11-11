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


def process_rep_data(df, dtw_k=3.0, stack_k=3, cluster_seconds=10, verbose=True):
    """
    Process REP data from MongoDB.
    
    Pipeline:
    1. Extract timestamp from _id
    2. Length-based cluster dropping (±10s, grouped by 'name')
    3. DTW outlier detection on 'data'
    4. Expand DTW outliers to time clusters (±10s, grouped by 'name')
    5. Extract geometric features from 'data'
    6. Stack by 'name' (material type)
    7. Extract feature vectors (geo_fv, fluct_fv)
    8. IQR outlier filtering by 'name' (material type)
    
    Works for ANY material: clay, deflocculant, water, etc.
    ALL original columns preserved.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from MongoDB with columns: _id, name, data, etc.
        'name' = material type (deflocculant, water, plaster, etc.)
    dtw_k : float
        DTW threshold multiplier (default: 3.0)
    stack_k : int
        Number of pulses per stack (default: 3)
    cluster_seconds : int
        Time window for cluster dropping (default: 10)
    verbose : bool
        Print progress (default: True)
        
    Returns:
    --------
    pd.DataFrame with all original columns plus:
        - data_stack: (k, m) stacked pulses
        - geom_stack: (k, 12) stacked features
        - geo_fv: (12,) geometric feature vector
        - fluct_fv: (72,) fluctuation feature vector
        - Time_Stamp: extracted timestamp
        
    Example:
    --------
    >>> df_final = process_rep_data(df)
    >>> df_final = process_rep_data(df, dtw_k=1.9, stack_k=5)
    """
    if verbose:
        print("="*60)
        print("REP DATA PROCESSING")
        print("="*60)
    
    df = df.copy()
    
    # Extract timestamp from MongoDB ObjectID
    if '_id' in df.columns and 'Time_Stamp' not in df.columns:
        if verbose:
            print("\n[Auto] Extracting timestamp from MongoDB ObjectID...")
        df = extract_timestamp_from_id(df)
    
    if 'Time_Stamp' not in df.columns:
        raise ValueError("No 'Time_Stamp' or '_id' column found")
    
    # [1/6] LENGTH-BASED CLUSTER DROPPING (grouped by 'name')
    if verbose:
        print(f"\n[1/6] Length-based cluster dropping (±{cluster_seconds}s by 'name')...")
    
    if not np.issubdtype(df['Time_Stamp'].dtype, np.datetime64):
        df['Time_Stamp'] = pd.to_datetime(df['Time_Stamp'])
    
    df = df.sort_values(['name', 'Time_Stamp']).reset_index(drop=True)
    
    df['rep_len'] = df['data'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else np.nan)
    modal_len = int(df['rep_len'].mode().iloc[0])
    
    odd_idx = df.index[df['rep_len'] != modal_len].tolist()
    
    drop_mask = pd.Series(False, index=df.index)
    cluster_radius = pd.Timedelta(seconds=cluster_seconds)
    
    for i in odd_idx:
        if drop_mask[i]:
            continue
        nm = df.at[i, 'name']
        ts = df.at[i, 'Time_Stamp']
        in_cluster = (
            (df['name'] == nm) &
            (df['Time_Stamp'].between(ts - cluster_radius, ts + cluster_radius))
        )
        drop_mask |= in_cluster
    
    df_clean = df.loc[~drop_mask].drop(columns=['rep_len']).reset_index(drop=True)
    
    if verbose:
        print(f"  Modal length: {modal_len}")
        print(f"  Dropped {drop_mask.sum()} rows, kept {len(df_clean)}")
    
    # [2/6] DTW OUTLIER DETECTION
    if verbose:
        print(f"\n[2/6] DTW outlier detection (k={dtw_k})...")
    
    dists, thr, keep = detect_outliers_dtw(df_clean, method="mad", k=dtw_k, verbose=verbose)
    
    # [3/6] EXPAND DTW OUTLIERS TO CLUSTERS (grouped by 'name')
    if verbose:
        print(f"\n[3/6] Expanding DTW outliers to clusters (±{cluster_seconds}s by 'name')...")
    
    bad = ~keep
    expanded_bad = expand_drop_to_clusters(df_clean, bad, seconds=cluster_seconds)
    
    df_filt = df_clean.loc[~expanded_bad].reset_index(drop=True)
    
    if verbose:
        print(f"  Expanded {bad.sum()} outliers to {expanded_bad.sum()}")
        print(f"  Kept {len(df_filt)} rows")
    
    # [4/6] EXTRACT GEOMETRIC FEATURES
    if verbose:
        print(f"\n[4/6] Extracting geometric features...")
    
    df_filt['vuong_sv'] = df_filt['data'].apply(extract_features)
    
    if verbose:
        print(f"  Extracted 12 features × {len(df_filt)} pulses")
    
    # [5/6] STACK (grouped by 'name' only)
    if verbose:
        print(f"\n[5/6] Stacking (k={stack_k} by 'name')...")
    
    keep_cols = [col for col in df_filt.columns if col not in ['vuong_sv']]
    
    out = []
    for name, g in df_filt.groupby('name', sort=False):
        g = g.reset_index(drop=True)
        num_stacks = len(g) // stack_k
        
        for i in range(num_stacks):
            stack = g.iloc[i*stack_k:(i+1)*stack_k]
            
            # Keep all columns from first row
            row = stack.iloc[0][keep_cols].to_dict()
            
            # Stack data and features
            row['data_stack'] = np.vstack(stack['data'].values)
            row['vuong_sv_stack'] = np.vstack(stack['vuong_sv'].values)
            
            out.append(row)
    
    stacked_df = pd.DataFrame(out)
    
    # Extract feature vectors
    stacked_df['geo_fv'] = stacked_df['vuong_sv_stack'].apply(
        lambda x: extract_features_nocv(x, "vuong")
    )
    stacked_df['fluct_fv'] = stacked_df['vuong_sv_stack'].apply(
        lambda x: extract_features_nocv(x, "matnoise")
    )
    
    stacked_df = stacked_df.rename(columns={'vuong_sv_stack': 'geom_stack'})
    
    if verbose:
        print(f"  Created {len(stacked_df)} stacks")
        data_len = len(df_filt['data'].iloc[0]) if len(df_filt) > 0 else 0
        print(f"  data_stack: ({stack_k}, {data_len})")
        print(f"  geom_stack: ({stack_k}, 12)")
        print(f"  geo_fv: (12,), fluct_fv: (72,)")
    
    # [6/6] IQR OUTLIER FILTERING (grouped by 'name')
    if verbose:
        print(f"\n[6/6] IQR outlier filtering by 'name' (material)...")
    
    df_final, _ = iqr_outlier_filter(stacked_df, verbose=verbose)
    
    if verbose:
        print("\n" + "="*60)
        print("COMPLETE!")
        print("="*60)
        print(f"Final shape: {df_final.shape}")
        print(f"Materials: {df_final['name'].unique().tolist()}")
    
    return df_final


def _as_1d(x):
    a = np.asarray(x, dtype=float)
    a = np.squeeze(a)
    if a.ndim > 1:
        a = a.ravel()
    if a.ndim != 1:
        raise ValueError(f"Sequence not 1-D after squeeze: shape={a.shape}")
    return a

def detect_outliers_dtw(df, data_col="data", method="mad", k=3.0, pct=90, verbose=True):
    """
    DTW outlier detection assuming equal-length 1-D sequences in df[data_col].
    Uses scalar-safe distance |a-b| to avoid scipy.euclidean 1-D checks.
    Returns: dists (np.ndarray), threshold (float), keep_mask (np.ndarray[bool])
    """
    # Canonicalize to 1-D arrays
    seqs = [ _as_1d(x) for x in df[data_col].values ]

    # Verify equal lengths
    lens = [len(s) for s in seqs]
    Lset = sorted(set(lens))
    if len(Lset) != 1:
        raise ValueError(f"Sequences not equal length: found lengths {Lset}. "
                         "Make them uniform first (trim/pad/drop).")

    # Reference = median curve
    arrs = np.vstack([s for s in seqs])  # (n, L)
    ref = np.median(arrs, axis=0)

    # Scalar-safe distance
    dist_scalar = lambda a, b: abs(a - b)

    # DTW distances
    dists = np.empty(len(seqs), dtype=float)
    for i, s in enumerate(seqs):
        d, _ = fastdtw(s, ref, dist=dist_scalar)
        dists[i] = d

    # Threshold
    if method == "mad":
        med = np.median(dists)
        mad = np.median(np.abs(dists - med)) + 1e-12
        thr = med + k * mad
    elif method == "percentile":
        thr = np.percentile(dists, pct)
    else:
        raise ValueError("method must be 'mad' or 'percentile'")

    keep = dists <= thr

    if verbose:
        print(f"DTW: kept {int(keep.sum())}/{len(keep)}, dropped {int((~keep).sum())}, threshold={thr:.3f}")

    return dists, thr, keep


def plot_pulse_by_name_sns(dataframe):
    unique_materials = dataframe['name'].unique().copy()

    for material in unique_materials:
        # Filter for this material only
        material_data = dataframe[dataframe['name'] == material].reset_index()#assigns index as a column now

        # Explode data into long-form
        long_df = (
            material_data[['index', 'data']]
            .explode('data')
            .assign(time_idx=lambda d: d.groupby('index').cumcount())  # x-axis
            #each element as its own row,same list means same index number.
            #but now the index of each element is distinguished by time_idx.
            .rename(columns={'index': 'trial'})
        )
        long_df['data'] = long_df['data'] / 1000  # Convert Pa to kPa

        # Plot using seaborn
        from matplotlib.cm import get_cmap
        from matplotlib.colors import to_hex,LinearSegmentedColormap

        unique_trials = long_df['trial'].unique()
        num_trials = len(unique_trials)

        # Define red → yellow → blue colormap
        tricolor_cmap = LinearSegmentedColormap.from_list("red_yellow_blue", ["#ff0000", "#ffff00", "#0000ff"], N=num_trials)
        colors = [to_hex(tricolor_cmap(i / (num_trials - 1))) for i in range(num_trials)]

        # Create palette mapping
        palette = dict(zip(unique_trials, colors))

        # Plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=long_df,
            x='time_idx',
            y='data',
            hue='trial',
            palette=palette,
            linewidth=1.5,
            alpha=0.9,
            hue_order=unique_trials
        )
        plt.title(f"Pulses for Material: {material}")
        plt.xlabel("Discrete-Time, index (n)")
        plt.ylabel("Pulse Value (KPa)")
        plt.legend(title='Trial', bbox_to_anchor=(1.15, 1), loc='upper left')
        plt.figure(figsize=(14, 8))  # more space for layout
        plt.tight_layout()
        plt.show()


def expand_drop_to_clusters(df, bad_mask, group_col="name", time_col="Time_Stamp", seconds=10):
    """
    Expand per-row 'bad' mask to include any rows within ±seconds
    of each bad row, but only within the same group (name).
    Returns an expanded boolean mask aligned to df.index.
    """
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

    expanded = pd.Series(False, index=df.index)
    delta = pd.Timedelta(seconds=seconds)

    bad_idx = np.where(bad_mask)[0]
    for i in bad_idx:
        nm = df.iloc[i][group_col]
        ts = df.iloc[i][time_col]
        in_cluster = (
            (df[group_col] == nm) &
            (df[time_col].between(ts - delta, ts + delta))
        )
        expanded |= in_cluster

    return expanded.values


def generate_stacked_dataset_triplets(df, k=3, group_cols=('name', 'clayBody'), include_weight=False):
    """
    Groups by `group_cols`, splits into triplets, and stacks:
      - 'vuong_sv' -> (k, n)
      - 'data' -> (k, m)
      - 'Relative_time_elapsed (s)' -> (k,)
      - 'Time_Elapsed (s)' -> (k,)              # global time
      - 'Trial' -> (k,)
    """
    print(f"Original dataset has {len(df)} rows.")
    out = []

    for gvals, g in df.groupby(list(group_cols), sort=False):
        g = g.reset_index(drop=True)
        num_triplets = len(g) // k

        for i in range(num_triplets):
            trip = g.iloc[i*k:(i+1)*k]

            row = {col: val for col, val in zip(group_cols, gvals)}
            row['vuong_sv_stack'] = np.vstack(trip['vuong_sv'].values)
            row['data_stack']     = np.vstack(trip['data'].values)
            
            if 'Relative_time_elapsed (s)' in trip.columns:
                row['time_stack_rel'] = trip['Relative_time_elapsed (s)'].to_numpy()
            
            if 'Time_Elapsed (s)' in trip.columns:
                row['time_stack_glb'] = trip['Time_Elapsed (s)'].to_numpy()
            
            if 'trial' in trip.columns:
                row['trial_stack']    = trip['trial'].to_numpy()
            
            if include_weight:
                if 'estimated_weight' in trip.columns:
                    row['avg_weight'] = trip['estimated_weight'].mean()
                
                loss_col = 'estimated_water_loss (g)' if 'estimated_water_loss (g)' in trip.columns else 'estimated_water_loss'
                if loss_col in trip.columns:
                    row['avg_water_loss'] = trip[loss_col].mean()

            out.append(row)

    new_df = pd.DataFrame(out)
    print(f"New dataset has {len(new_df)} triplets after stacking.")
    return new_df

def triplets_extract_features(df_clean_filt, k=3, group_cols=["name", "clayBody"],include_weight=False):
    if include_weight == False:
        stacked_df = generate_stacked_dataset_triplets(df_clean_filt, k=k, group_cols=group_cols, include_weight=False)
    else:
        stacked_df = generate_stacked_dataset_triplets(df_clean_filt, k=k, group_cols=group_cols , include_weight=True)
        
    # 1. Apply both feature extraction functions first
    stacked_df['vuong_fv'] = stacked_df['vuong_sv_stack'].apply(
        lambda x: extract_features_nocv(x, "vuong")
    )

    stacked_df['matnoise_fv'] = stacked_df['vuong_sv_stack'].apply(
        lambda x: extract_features_nocv(x, "matnoise")
    )

    # # 3. Now it's safe to check shapes
    stacked_df['vuong_fv'][0].shape, stacked_df['matnoise_fv'][0].shape
    print(stacked_df)

    # Suppose your feature column is named "matnoise_fv" and stores arrays
    has_nan = stacked_df["matnoise_fv"].apply(lambda arr: np.isnan(arr).any())

    print("Rows with NaN in matnoise_fv:", has_nan.sum())

    return stacked_df


def iqr_outlier_filter_grouped(df, group_col, colnames, verbose=True):
    """
    Detect and remove outliers based on cosine distance + IQR,
    applied separately per group (e.g., per clayBody).

    Args:
        df: pandas DataFrame
        group_col: column name to group by (e.g., "clayBody")
        colnames: list of feature-vector columns (np.array)
        verbose: print debug info if True

    Returns:
        df_clean: DataFrame with outliers removed
        outlier_info: dict[group][col] = thresholds and counts
    """
    outlier_info = {}
    global_mask = pd.Series([False] * len(df), index=df.index)

    for group, subdf in df.groupby(group_col):
        outlier_info[group] = {}
        if verbose:
            print(f"=== Group: {group} (n={len(subdf)}) ===")
        mask = pd.Series([False] * len(subdf), index=subdf.index)

        for col in colnames:
            X = np.stack(subdf[col].values)
            centroid = X.mean(axis=0, keepdims=True)
            dists = cosine_distances(X, centroid).ravel()

            # IQR thresholds
            q1, q3 = np.percentile(dists, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = (dists < lower) | (dists > upper)

            mask = mask | outliers

            outlier_info[group][col] = {
                "q1": q1, "q3": q3, "iqr": iqr,
                "lower": lower, "upper": upper,
                "n_total": len(dists),
                "n_outliers": int(outliers.sum())
            }

            if verbose:
                print(f"[{col}] Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
                print(f"       Thresholds: [{lower:.4f}, {upper:.4f}]")
                print(f"       Outliers: {outliers.sum()} / {len(dists)}")

        global_mask.loc[mask.index] = global_mask.loc[mask.index] | mask

    df_clean = df.loc[~global_mask].copy()

    if verbose:
        print(f"\n[Summary] Removed {global_mask.sum()} total outliers "
              f"across {len(colnames)} columns and {df[group_col].nunique()} groups.")
        print(f"Remaining samples: {len(df_clean)} / {len(df)}")

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
