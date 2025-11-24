# cleaning.py
import numpy as np
import pandas as pd
from datetime import datetime
from fastdtw import fastdtw
from sklearn.metrics.pairwise import cosine_distances
from sklearn.utils import resample
import warnings

# Import only what you need from rep and viz
from rep import filter_pulse, extract_features, extract_features_nocv
from viz import plot_pulse_by_name_sns, plot_lines_by_condition

warnings.filterwarnings('ignore')

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


def stack_column(values):
    """
    Stack or aggregate a column of triplet values based on inferred type.
    """
    first = values[0]

    # 1. Array-like (list or ndarray)
    if isinstance(first, (list, np.ndarray)):
        arrs = [np.asarray(v) for v in values]
        return np.vstack(arrs)

    # 2. Numeric scalar
    if np.isscalar(first) and isinstance(first, (int, float, np.number)):
        return float(np.mean(values))

    # 3. Everything else: strings, categories, objects
    # Keep as vector of length k
    return np.array(values, dtype=object)


def stackify(df, k=3, group_cols=("clayBody",), feature_source_col="vuong_sv",
             extract_geom=True, extract_fluct=True):
    """
    Generalized k-stacking function.

    Groups by group_cols, forms windows of size k in order of timestamp,
    stacks all non-group columns automatically, then optionally extracts
    geometric and fluctuation features.

    Args:
        df : cleaned dataframe with feature_source_col already computed
        k : window size
        group_cols : tuple of columns used to form groups
        feature_source_col : column containing pulse vectors to extract features from
        extract_geom : extract geom features using extract_features_nocv
        extract_fluct : extract fluctuation features using extract_features_nocv

    Returns:
        stacked_df : k-stacked dataframe with *_stack and feature columns
    """

    df = df.copy()
    print(f"Stackify: input rows = {len(df)}, grouping by {group_cols}, k={k}")

    # Identify data columns (everything not used for grouping)
    data_cols = [c for c in df.columns if c not in group_cols]

    out_rows = []

    # Group by the group_cols
    for gvals, g in df.groupby(list(group_cols), sort=False):
        g = g.sort_values("time_stamp").reset_index(drop=True)

        num_windows = len(g) // k
        for i in range(num_windows):
            block = g.iloc[i*k:(i+1)*k]

            row = {col: val for col, val in zip(group_cols, gvals)}

            # Stack each data column
            for col in data_cols:
                vals = block[col].tolist()
                row[f"{col}_stack"] = stack_column(vals)

            out_rows.append(row)

    stacked_df = pd.DataFrame(out_rows)
    print(f"Stackify: produced {len(stacked_df)} stacked rows")

    # Feature extraction
    if extract_geom:
        stacked_df["geom_fv"] = stacked_df[f"{feature_source_col}_stack"].apply(
            lambda x: extract_features_nocv(x, "vuong")
        )

    if extract_fluct:
        stacked_df["fluctuation_fv"] = stacked_df[f"{feature_source_col}_stack"].apply(
            lambda x: extract_features_nocv(x, "matnoise")
        )

    # Diagnostics
    if extract_fluct:
        num_nan = stacked_df["fluctuation_fv"].apply(lambda arr: np.isnan(arr).any()).sum()
        print(f"Stackify: fluctuation_fv rows with NaN = {num_nan}")

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
    """
    Generates:
      time_stamp
      time_elapsed_s
      relative_time_elapsed_s
    """
    df = df.copy()

    # Convert Mongo ObjectId to datetime
    df['time_stamp'] = [
        datetime.fromtimestamp(int(str(oid)[:8], 16)) for oid in df['_id']
    ]

    # Global elapsed seconds
    df['time_elapsed_s'] = (
        df['time_stamp'] - df['time_stamp'].min()
    ).dt.total_seconds()

    # Relative time per name
    df['relative_time_elapsed_s'] = (
        df.groupby('name')['time_stamp']
            .transform(lambda s: (s - s.min()).dt.total_seconds())
    )

    df = df.sort_values(
        by=["name", "relative_time_elapsed_s"]
    ).reset_index(drop=True)

    return df


def normalize_time_columns(df):
    """
    Ensures presence of:
      time_stamp
      time_elapsed_s
      relative_time_elapsed_s

    If _id exists and the fields are missing, they are created.
    """
    df = df.copy()

    needs_gen = (
        '_id' in df.columns
        and (
            'time_stamp' not in df.columns
            or 'relative_time_elapsed_s' not in df.columns
        )
    )

    if needs_gen:
        df = generate_time_stamp(df)

    # Ensure correct type and order
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df = df.sort_values(
        ['name', 'time_stamp']
    ).reset_index(drop=True)

    return df


def finalize_generic_target(df):
    return df.rename(columns={
        "data_stack": "REP_pulses",
        "vuong_sv_stack": "geom_stack",
        "vuong_fv": "geom_fv",
        "matnoise_fv": "fluctuation_fv"
    })







def filter_by_rep_length(df, cluster_seconds=10):
    """
    Removes rows whose REP length does not match the modal length, and
    expands removals to all rows within +/- cluster_seconds for the same name.

    Returns:
        cleaned_df, info_dict
    """
    df = df.copy()

    # Compute REP lengths
    df['rep_len'] = df['data'].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else np.nan
    )
    modal_len = int(df['rep_len'].mode().iloc[0])

    # Identify rows with non modal length
    odd_idx = df.index[df['rep_len'] != modal_len]

    # Cluster expansion mask
    drop_mask = pd.Series(False, index=df.index)
    for i in odd_idx:
        nm = df.at[i, 'name']
        ts = df.at[i, 'time_stamp']   # lowercase version
        cluster = (
            (df['name'] == nm)
            & df['time_stamp'].between(
                ts - pd.Timedelta(seconds=cluster_seconds),
                ts + pd.Timedelta(seconds=cluster_seconds)
            )
        )
        drop_mask |= cluster

    # Collect rows that will be dropped
    dropped_rows = df.loc[drop_mask, ['name', 'rep_len']]
    per_name = dropped_rows['name'].value_counts()

    n_drop = int(drop_mask.sum())
    n_keep = len(df) - n_drop

    # Informative messages
    print(f"Modal REP length is {modal_len}")
    print(f"Rows with non modal length: {len(odd_idx)}")
    print(f"Total rows dropped after cluster expansion: {n_drop}")
    print(f"Rows kept: {n_keep}")

    if len(per_name) > 0:
        print("Dropped rows per name:")
        print(per_name)
    else:
        print("No rows dropped. All REPs matched the modal length.")

    # Apply removal
    cleaned_df = (
        df.loc[~drop_mask]
        .drop(columns=['rep_len'])
        .reset_index(drop=True)
    )

    # Metadata for logs and audits
    info = {
        "modal_len": modal_len,
        "num_dropped": n_drop,
        "num_kept": n_keep,
        "dropped_per_name": per_name.to_dict(),
        "dropped_indices": dropped_rows.index.to_list()
    }

    return cleaned_df, info


def filter_dtw_outliers(df, target="name", DTW_graph=False):
    dists, thr, keep = detect_outliers_dtw(df, data_col="data", method="mad", k=3)
    bad = ~keep
    expanded = expand_drop_to_clusters(df, bad, group_col=target, time_col="time_stamp", seconds=10)
    df = df.loc[~expanded].reset_index(drop=True)

    if DTW_graph:
        plot_pulse_by_name_sns(df, target=target)

    return df

def harmonize_arrays(df, expected_len):
    """
    Filters rows where data arrays do not match expected_len.
    Converts remaining data arrays to numpy arrays.

    Returns:
        cleaned_df, info_dict
    """
    df = df.copy()

    # Compute lengths
    lengths = df['data'].apply(lambda x: len(x) if hasattr(x, '__len__') else np.nan)
    df['data_len'] = lengths

    # Show length distribution for debugging
    print("Array length distribution before filtering:")
    print(lengths.value_counts().sort_index(), "\n")

    # Filter rows
    keep_mask = lengths == expected_len
    dropped = df.loc[~keep_mask]
    kept = df.loc[keep_mask]

    # Informative messaging
    print(f"Expected array length: {expected_len}")
    print(f"Total rows: {len(df)}")
    print(f"Rows kept: {len(kept)}")
    print(f"Rows dropped: {len(dropped)}")

    if len(dropped) > 0:
        print("\nDropped lengths:")
        print(dropped['data_len'].value_counts().sort_index())

        print("\nExamples of dropped rows:")
        print(dropped[['name', 'data_len']].head())

    else:
        print("All rows have the expected array length.")

    # Clean up
    kept = kept.drop(columns=['data_len']).reset_index(drop=True)
    kept['data'] = kept['data'].apply(np.asarray)

    info = {
        "expected_len": expected_len,
        "num_kept": len(kept),
        "num_dropped": len(dropped),
        "dropped_lengths": dropped['data_len'].value_counts().to_dict(),
        "dropped_indices": dropped.index.to_list()
    }

    return kept, info


def extract_all_features(df):
    df = df.copy()
    df['vuong_clean'] = df['data'].apply(filter_pulse)
    df['vuong_sv'] = df['data'].apply(extract_features)
    return df




def balancing_function(df, target, df_balancing=False, min_class_size=10,
                       n_bins=10, bin_strategy="quantile"):
    """
    Balancing helper for continuous float target.

    Behavior:
    - Converts TARGET into discrete bins.
    - If df_balancing=False:
        -> prints bin counts and returns df unchanged.
    - If df_balancing=True:
        -> removes bins with < min_class_size samples,
           down-samples all remaining bins to the same size.
    """

    df = df.copy()

    # -----------------------------------------------
    # 1. Create bins from the continuous target
    # -----------------------------------------------
    if bin_strategy == "uniform":
        df["label_bin"] = pd.cut(df[target], bins=n_bins)
    else:  # quantile-based (equal-sized bins)
        df["label_bin"] = pd.qcut(df[target], q=n_bins, duplicates="drop")

    label_col = "label_bin"

    class_counts = df[label_col].value_counts().sort_index()
    print("\n=== Bin counts ===")
    print(class_counts)

    # -------------------------------------------------
    # 2) No balancing requested → return as is
    # -------------------------------------------------
    if not df_balancing:
        print("\n[INFO] df_balancing=False → no filtering or balancing applied.")
        return df.reset_index(drop=True)

    # -------------------------------------------------
    # 3) Filter bins that do not meet min_class_size
    # -------------------------------------------------
    valid_bins = class_counts[class_counts >= min_class_size].index.tolist()
    dropped = set(class_counts.index) - set(valid_bins)

    print(f"\nDropping bins with < {min_class_size} samples:")
    print(dropped)

    df_filtered = df[df[label_col].isin(valid_bins)].copy()

    print("\n=== Bin counts after filtering ===")
    print(df_filtered[label_col].value_counts().sort_index())

    # If fewer than 2 bins remain, no balancing possible
    if len(valid_bins) < 2:
        print("\n[WARNING] <2 valid bins → returning filtered dataset only.")
        return df_filtered.reset_index(drop=True)

    # -------------------------------------------------
    # 4) Uniform sampling across bins
    # -------------------------------------------------
    min_size = df_filtered[label_col].value_counts().min()
    balanced_chunks = []

    for lbl, subset in df_filtered.groupby(label_col):
        subset_bal = resample(
            subset,
            replace=False,
            n_samples=min_size,
            random_state=42
        )
        balanced_chunks.append(subset_bal)

    df_balanced = pd.concat(balanced_chunks).reset_index(drop=True)

    print("\n=== Bin counts after balancing ===")
    print(df_balanced[label_col].value_counts().sort_index())

    return df_balanced


def clean_data_master(df, TARGET, head=5, use_calibration = False, DTW_graph=False, df_balancing=False, bins=5, bin_strategy="uniform"):
    if df is None or len(df) == 0:
        raise ValueError("Input dataframe is empty")

    print("\nComputing timestamps")
    print("-"*20)
    df = normalize_time_columns(df)
    
    print("\nValidating REP Length")
    print("-"*20)
    df, _ = filter_by_rep_length(df)

    if use_calibration:
        print("\nApplying Global Calibration (Batch ≠ 1)")
        print("-"*40)
        calibrator = load_default_calibrator()
    
        df = df.copy()
        for idx, row in df.iterrows():
            if int(row["batch"]) != 1:
                df.at[idx, "data"] = calibrator.apply(
                    np.array(row["data"])
                )
    else:
        print("\n(No calibration applied)")

    print("\nREP Outlier Detection (DTW x MAD)")
    print("-"*20)
    df = filter_dtw_outliers(df, DTW_graph=DTW_graph, target=TARGET)
    
    print("\nValidate REP Length # 2")
    print("-"*20)
    df, _= harmonize_arrays(df, expected_len=22)
    
    print("\nExtracting Features")
    print("-"*20)
    df = extract_all_features(df)
    
    print("\nStacking")
    print("-"*20)
    stacked_df = stackify(
        df,
        k=3,
        group_cols=("clayBody", TARGET),
        feature_source_col="vuong_sv"
    )
    stacked_df = finalize_generic_target(stacked_df)


    print("\nFluctuation Feature Outlier Detection")
    print("-"*20)
    df_clean, outlier_info = iqr_outlier_filter_grouped(stacked_df, "clayBody", ["geom_fv", "fluctuation_fv"], verbose=True)


    print("\nBalancing")
    df_clean = balancing_function(
        df_clean,
        target=TARGET,
        df_balancing=df_balancing,
        n_bins=bins,
        bin_strategy=bin_strategy
    )

    return df_clean, _
