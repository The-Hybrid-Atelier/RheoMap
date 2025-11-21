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


def plot_pulse_by_name_sns(
        df, 
        target="name",
        mode="each",          
        scale=1000.0,         
        figsize=(8, 4),
        linewidth=4,
        alpha=0.9,
        font_scale=1):
    """
    High-quality research visualization for REP pulses with two modes:
        - each: plot each pulse individually
        - mean: plot mean curve (+/- SE)
    """

    sns.set(style="whitegrid")
    sns.set_context("talk", font_scale=font_scale)

    unique_groups = df[target].unique()

    for group in unique_groups:
        group_df = df[df[target] == group].reset_index(drop=True)

        # ---- Explode pulses into long format ----
        long_df = (
            group_df
            .reset_index()
            .rename(columns={"index": "trial"})
            .explode("data")
            .assign(
                time_idx=lambda d: d.groupby("trial").cumcount(),
                data=lambda d: d["data"].astype(float) / scale
            )
        )

        # ---- Choose a nicer ordinal palette ----
        unique_trials = long_df["trial"].unique()
        num_trials = len(unique_trials)

        # Smooth sequential palette for ordinal trial values
        palette = sns.color_palette("crest", n_colors=num_trials)

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=figsize)

        if mode == "each":
            sns.lineplot(
                data=long_df,
                x="time_idx",
                y="data",
                hue="trial",
                linewidth=linewidth,
                alpha=alpha,
                ax=ax,
                palette=palette,
                hue_order=unique_trials
            )
            ax.set_title(f"Pulses for: {group}", fontsize=20)

        elif mode == "mean":
            stats = (
                long_df
                .groupby("time_idx")["data"]
                .agg(["mean", "sem"])
                .reset_index()
            )

            # SE band
            ax.fill_between(
                stats["time_idx"],
                stats["mean"] - stats["sem"],
                stats["mean"] + stats["sem"],
                color="tab:blue",
                alpha=0.25,
                linewidth=0
            )

            sns.lineplot(
                data=stats,
                x="time_idx",
                y="mean",
                linewidth=linewidth + 1.5,
                color="tab:blue",
                ax=ax
            )
            ax.set_title(f"Mean Pulse (±SE) for: {group}", fontsize=20)

        else:
            raise ValueError("mode must be 'each' or 'mean'")

        # ---- Labels and styling ----
        ax.set_xlabel("Discrete Time Index (n)", fontsize=18)
        ax.set_ylabel("Air Pressure (kPa)", fontsize=18)

        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.25)

        # ---- Always put legend to the right ----
        if mode == "each":
            ax.legend(
                title="Trial",
                fontsize=12,
                title_fontsize=14,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0.0
            )
        else:
            ax.legend(
                ["Mean ± SE"],
                fontsize=12,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0.0
            )

        fig.tight_layout()
        plt.show()
