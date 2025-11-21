# viz.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import feature helpers if needed
from rep import filter_pulse, extract_features

# ============================================================================
#    VISUALIZATION HELPERS
# ============================================================================



# REP (Retraction-Extrusion Pulse) 
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

def plot_pulse_by_name_sns(dataframe, target="name"):
    unique_materials = dataframe[target].unique().copy()

    for material in unique_materials:
        # Filter for this material only
        material_data = dataframe[dataframe[target] == material].reset_index()#assigns index as a column now

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
