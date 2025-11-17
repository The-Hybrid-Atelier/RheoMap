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
from sklearn.utils import resample

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
    smoothed = np.convolve(pulse, np.ones(window_size) / window_size, mode='valid')
    # Pad to match original length by repeating edge values
    pad_left = (window_size - 1) // 2
    pad_right = window_size - 1 - pad_left
    return np.pad(smoothed, (pad_left, pad_right), mode='edge')

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
    pulse_med = median_filter(pulse, kernel_size=median_k)

    # Step 2: Wavelet denoising
    pulse_wave = wavelet_filter(pulse_med, wavelet=wavelet, level=level)
    
    # Match original length in case waverec adds samples
    pulse_wave = pulse_wave[:len(pulse_med)]

    # Step 3: Moving average smoothing
    pulse_smooth = smooth(pulse_wave, window_size=window_size)

    return pulse_smooth

def extract_features(pulse):
    """
    Extract geometric features from a pulse signal.
    Automatically applies filtering (median + wavelet + smoothing) before extraction.
    
    Parameters:
    -----------
    pulse : array-like
        Raw pulse signal
        
    Returns:
    --------
    list : 12 extracted features
        [basevalue, min_retraction_value, max_extrusion_value, equilibrium_value,
         max_retraction_time, max_extrusion_time, equilibrium_time, 
         extrusion_period, equilibrium_period, fluid_release_point_time,
         fluid_release_point_value, fluid_release_point_period]
    """
    # Always apply filtering before feature extraction
    
    # pulse = filter_pulse(pulse) #### PRE FILTERING - SINGLE REP FILTERING
    
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


# ============================================================================
#    AFTER RHEOMAP
#    HELPER FUNCTIONS FOR PROCESS_REP_DATA 
# ============================================================================

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

def audit_reps(df, data_col="data", group_col="name", head=5):
    # lengths per row
    lengths = df[data_col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else np.nan)
    has_nan = df[data_col].apply(lambda x: np.any(np.isnan(x)) if isinstance(x, (list, np.ndarray)) else True)

    out = pd.DataFrame({
        "len": lengths,
        "has_nan": has_nan,
        group_col: df[group_col].values
    }, index=df.index)

    print("=== Overall ===")
    print("Rows:", len(out))
    print("Unique lengths:", sorted(out['len'].dropna().unique().astype(int)))
    print("Rows with NaNs:", int(out['has_nan'].sum()))
    print()

    print("=== By group (length stats) ===")
    print(out.groupby(group_col)['len'].agg(['count','min','max','median','nunique']).sort_values(['nunique','max','min'], ascending=False))
    print()

    print("=== By group (NaN counts) ===")
    print(out.groupby(group_col)['has_nan'].sum().rename('nan_rows'))
    print()

    # show examples of problematic rows
    bad_len_mask = out['len'] != out['len'].mode().iloc[0]  # not the modal length
    bad_nan_mask = out['has_nan']
    bad_idx = out[bad_len_mask | bad_nan_mask].index.tolist()[:head]
    if bad_idx:
        print(f"Examples of problematic rows (first {len(bad_idx)}):", bad_idx)
    else:
        print("No problematic rows found.")

    return out

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
    
def balancing_function(df, TARGET):
    if TARGET == "clayBody":
        # Find the smallest class size
        class_counts = df["clayBody"].value_counts()
        min_size = class_counts.min()
        print("Class counts before balancing:\n", class_counts)
        
        balanced_parts = []
        for clay, subset in df.groupby("clayBody"):
            subset_bal = resample(
                subset,
                replace=False,
                n_samples=min_size,
                random_state=42
            )
            balanced_parts.append(subset_bal)
        
        df_balanced = pd.concat(balanced_parts).reset_index(drop=True)
        print("Class counts after balancing:\n", df_balanced["clayBody"].value_counts())
        return df_balanced
    #if TARGET == something else
    else:
        # Return unbalanced data if TARGET doesn't match
        print(f"WARNING: Balancing requested but TARGET='{TARGET}' doesn't match 'clayBody'. Returning unbalanced data.")
        return df
        
# ============================================================================
# MASTER FUNCTION
# ============================================================================

def clean_data_master(df, TARGET, head=5, DTW_graph=False, df_balancing=False, 
                      balance_strategy='ensemble', balance_target=None):
    """
    Master function to clean and process REP sensor data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with required columns: 'data', 'name', 'Time_Stamp'
    TARGET : str
        One of: 'time', 'mixing', or other (determines output column naming)
    head : int
        Number of examples to show in audit (default: 5)
    DTW_graph : bool
        Whether to plot DTW graphs (default: False)
    df_balancing : bool
        Whether to apply class balancing (default: False)
    balance_strategy : str
        Balancing strategy (default: 'ensemble')
        Options: 'ensemble', 'tomek', 'enn', 'ncr', 'undersample_smart',
                 'hybrid', 'smote', 'oversample_all', 'undersample_all'
    balance_target : int
        Target samples per class for 'hybrid' strategy (default: 100)
        
    Returns:
    --------
    df_clean : pd.DataFrame
        Cleaned dataframe with features
    outlier_info : dict
        Information about outliers removed
    """
    # Input validation
    if df is None or len(df) == 0:
        raise ValueError("Input dataframe is empty")
    
    # Generate time-related columns if needed
    # This is required for TARGET='time' to work properly
    if '_id' in df.columns:
        # If Time_Stamp doesn't exist, or if we need time-related columns
        if 'Time_Stamp' not in df.columns or 'Relative_time_elapsed (s)' not in df.columns:
            df = generate_time_stamp(df)
            print("Generated Time_Stamp and time-related columns from MongoDB _id")
    
    audit = audit_reps(df, data_col="data", group_col="name")

    modal_len = 22
    # Boolean mask for rows that are not 22
    mask_weird = audit['len'] != modal_len
    # Count them
    num_weird = mask_weird.sum()
    print(f"Weird rows (length not {modal_len}):", num_weird)
    # Show breakdown by group
    print(audit[mask_weird].groupby('name')['len'].value_counts())

    #Dropping clusters if they have different lengths
    # main_df has columns: 'name', 'Time_Stamp' (datetime-like), 'data' (1D array/list)
    # 0) Ensure types & order
    main_df = df.copy()
    if not np.issubdtype(main_df['Time_Stamp'].dtype, np.datetime64):
        main_df['Time_Stamp'] = pd.to_datetime(main_df['Time_Stamp'])
    main_df = main_df.sort_values(['name', 'Time_Stamp']).reset_index(drop=True)

    # 1) Compute REP lengths and modal (expected) length
    main_df['rep_len'] = main_df['data'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else np.nan)
    modal_len = int(main_df['rep_len'].mode().iloc[0])

    # 2) Indices of odd-length rows
    odd_idx = main_df.index[main_df['rep_len'] != modal_len].tolist()

    # 3) Build a drop mask by expanding each odd row to its 10s cluster (same 'name')
    drop_mask = pd.Series(False, index=main_df.index)
    cluster_radius = pd.Timedelta(seconds=10)

    for i in odd_idx:
        if drop_mask[i]:  # already covered by a previous cluster
            continue
        nm = main_df.at[i, 'name']
        ts = main_df.at[i, 'Time_Stamp']
        # Same name, within ±10s of this REP
        in_cluster = (
            (main_df['name'] == nm) &
            (main_df['Time_Stamp'].between(ts - cluster_radius, ts + cluster_radius))
        )
        drop_mask |= in_cluster  # union of all cluster members

    dropped_rows = main_df.loc[drop_mask, ["name", "Time_Stamp", "rep_len"]]
    print(dropped_rows.sort_values(["name", "Time_Stamp"]))

    # 4) Apply drop
    df_clean = main_df.loc[~drop_mask].drop(columns=['rep_len']).reset_index(drop=True)

    # 5) Report
    n_drop = int(drop_mask.sum())
    n_keep = len(main_df) - n_drop
    per_name = main_df.loc[drop_mask, 'name'].value_counts()

    print(f"Modal REP length = {modal_len}")
    print(f"Dropped {n_drop} rows in {per_name.size} cluster(s); kept {n_keep} rows.")
    print("Dropped per name:")
    print(per_name)


    #before
    print("Data Before DTW: ")
    if DTW_graph == True:
        plot_pulse_by_name_sns(df_clean)
    print(df_clean['name'])
    dists, thr, keep = detect_outliers_dtw(
        df_clean, data_col="data",
        method="mad", k=3 # for everything else is k = 3, except mixing k = 1.9 (visually)
    )

    # Expand drops to clusters (±10 s within same name)
    bad = ~keep
    expanded_bad = expand_drop_to_clusters(df_clean, bad, group_col="name",
                                        time_col="Time_Stamp", seconds=10)
    # Final filtered DF
    df_clean_filt = df_clean.loc[~expanded_bad].reset_index(drop=True)
    if DTW_graph == True:
        plot_pulse_by_name_sns(df_clean_filt)
    print("Data After DTW: ")
    if 'samplingLocation' in df_clean_filt.columns:
        print(df_clean_filt.groupby('samplingLocation').describe())


    df_clean_filt['name'] = ["mixed-5mins" if name == 'mixed-5min' else name for name in df_clean_filt['name'] ]
    # 1. Check that all items in the 'data' column are arrays
    all_are_arrays = df_clean_filt['data'].apply(lambda x: isinstance(x, (np.ndarray, list))).all()

    # 2. Check that all arrays are the same length and filter to length 22
    array_lengths = df_clean_filt['data'].apply(len)
    df_clean_filt = df_clean_filt[array_lengths == 22].reset_index(drop=True)
    
    # 3. Re-check after filtering
    array_lengths_after = df_clean_filt['data'].apply(len)
    all_same_length = array_lengths_after.nunique() == 1

    if all_same_length and all_are_arrays:
        # Convert to numpy arrays
        df_clean_filt['data'] = df_clean_filt['data'].apply(np.array)
        print("Converted to numpy arrays")
        # Summary
        print(f"All data entries are arrays/lists? {all_are_arrays}")
        print(f"All data arrays have the same length? {all_same_length}")
        if all_same_length:
            print(f"Array length: {array_lengths_after.iloc[0]}")
        else:
            print("Lengths found:", array_lengths_after.value_counts())

    # Description
    name_summary = df_clean_filt.groupby('name').size().reset_index(name='count').sort_values(by='count', ascending=False)
    print(name_summary)

    #cleaned and Raw data, we use Raw since it has more variation
    fig, ax = plot_lines_by_condition(df_clean_filt, pulse_col="data", condition_col="name")
    ax.set_title("RAW", fontsize=18)

    df_clean_filt_cleaned = df_clean_filt.copy()
    df_clean_filt_cleaned['vuong_clean'] = df_clean_filt['data'].apply(filter_pulse)
    fig, ax = plot_lines_by_condition(df_clean_filt_cleaned, pulse_col="vuong_clean", condition_col="name")
    ax.set_title("CLEANED", fontsize=18)


    # Generate feature vectors from raw data
    df_clean_filt['vuong_sv'] = df_clean_filt['data'].apply(extract_features)

    # Validate clayBody column exists for stacking
    if 'clayBody' not in df_clean_filt.columns:
        raise ValueError("'clayBody' column is required for stacking and feature extraction")
    
    stacked_df = triplets_extract_features(df_clean_filt, include_weight = False)

    if TARGET == 'time':
        renamed_df = stacked_df.rename(columns={
            "name": "trial_identifier",
            "data_stack": "REP_pulses",
            "vuong_sv_stack": "geom_stack",
            "vuong_fv": "geom_fv",
            "matnoise_fv": "fluctuation_fv",
            'time_stack_rel':'time_stack_rel'
        })

        # Reorder to your requested layout
        renamed_df = renamed_df[[
            "clayBody",
            "trial_identifier",
            "REP_pulses",
            "geom_stack",
            "geom_fv",
            "fluctuation_fv",
            "time_stack_rel"
        ]]
        renamed_df['dwell_time_min'] = renamed_df.groupby('trial_identifier').cumcount()

    elif TARGET == 'mixing':
        renamed_df = stacked_df.rename(columns={
            "name": "time_since_mixed_min",
            "data_stack": "REP_pulses",
            "vuong_sv_stack": "geom_stack",
            "vuong_fv": "geom_fv",
            "matnoise_fv": "fluctuation_fv"
        })

        # Reorder to your requested layout
        renamed_df = renamed_df[[
            "clayBody",
            "time_since_mixed_min",
            "REP_pulses",
            "geom_stack",
            "geom_fv",
            "fluctuation_fv"
        ]]
    else:
        renamed_df = stacked_df.rename(columns={
            "data_stack": "REP_pulses",
            "vuong_sv_stack": "geom_stack",
            "vuong_fv": "geom_fv",
            "matnoise_fv": "fluctuation_fv"
        })
        renamed_df = renamed_df[[
            "clayBody",
            "name",
            "REP_pulses",
            "geom_stack",
            "geom_fv",
            "fluctuation_fv"
        ]]

    has_nan = renamed_df["fluctuation_fv"].apply(lambda arr: np.isnan(arr).any())
    print("Rows with NaN in fluctuation_fv:", has_nan.sum())

    # Final IQR outlier filtering
    df_clean, outlier_info = iqr_outlier_filter_grouped(
        renamed_df, group_col="clayBody",
        colnames=["fluctuation_fv", "geom_fv"],
        verbose=True
    )
    
    if len(df_clean) == 0:
        raise ValueError("All data was filtered out! No samples remaining after cleaning pipeline.")
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: {len(df_clean)} samples ready for ML")
    print(f"{'='*60}\n")

    if df_balancing == True:
        df_clean = balancing_function(df_clean,TARGET)
    
    return df_clean, outlier_info
