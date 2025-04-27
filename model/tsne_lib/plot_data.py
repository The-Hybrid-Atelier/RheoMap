from load_data import *
from test_data import FEATURES
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np

custom_palette = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#6A5ACD', '#FF1493']  # Tomato, SteelBlue, LimeGreen, Gold, SlateBlue, DeepPink
path = "/workspaces/RheoPulse/model/tsne_lib/plot"
materialnames = ["honey", "ketchup", "oatmeal", "water"] # materialnames (list): List of material names to include in the plot.

########### Plot functions ############
def box_plot_feature(scaled_data, material_name):
    """
    Creates three separate vertical box plots for features grouped by 
    '_value', '_period', and '_time', comparing scaled_data and master_data
    for the same material. Only displays the single data point from 
    'ThingPlus 1 Data Sample' on the box plot of 'PA Data' and adds a
    legend for the single data point in all three box plots.

    Args:
        scaled_data (DataFrame): DataFrame containing scaled data.
    """
    # Set Seaborn theme
    sns.set_theme(style="ticks")

    # Load master data and extract the material name from scaled_data
    master_data = load_masterdata()

    # Filter master_data to include only the rows with the specified material
    filtered_master_data = master_data[master_data['material'] == material_name]

    # Group features by their suffix
    value_features = [f for f in FEATURES if f.endswith('_value')]
    period_features = [f for f in FEATURES if f.endswith('_period')]
    time_features = [f for f in FEATURES if f.endswith('_time')]

    # Function to create combined data for a group of features
    def create_combined_data(features):
        combined_data = pd.DataFrame({
            'Feature': [],
            'Value': [],
            'Dataset': []
        })
        for feature in features:
            if feature in scaled_data.columns and feature in master_data.columns:
                combined_data = pd.concat([
                    combined_data,
                    pd.DataFrame({
                        'Feature': [feature] * len(filtered_master_data[feature]),
                        'Value': filtered_master_data[feature],
                        'Dataset': ['PA Data'] * len(filtered_master_data[feature])
                    })
                ], ignore_index=True)
        return combined_data

    # Custom color palette for 'PA Data'
    palette = {'PA Data': 'lightgrey'}

    # Extract the single data point values for all features
    single_data_point_values = {feature: scaled_data[feature].iloc[0] for feature in FEATURES}

    # Plot each group of features with x-axis as Feature and y-axis as Value
    plt.figure(figsize=(14, 18))

    # Helper function to plot the box plot and overlay scatter points
    def plot_feature_group(features, combined_data, title, subplot_index):
        plt.subplot(3, 1, subplot_index)
        sns.boxplot(data=combined_data, x='Feature', y='Value', hue='Dataset', palette=palette)

        # Overlay the single data point
        for i, feature in enumerate(features):
            if feature in single_data_point_values:
                plt.scatter(
                    x=i,
                    y=single_data_point_values[feature],
                    color='red', s=100, edgecolor='black', zorder=10
                )

        # Add the legend to each plot
        plt.scatter([], [], color='red', s=100, edgecolor='black', label='ThingPlus 1 Data Sample')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title(f"{title} ({material_name})")
        plt.xticks(range(len(features)), features, rotation=45)

    # Plot for '_value' features
    combined_data_value = create_combined_data(value_features)
    plot_feature_group(value_features, combined_data_value, "Box Plots for Value Features", 1)

    # Plot for '_period' features
    combined_data_period = create_combined_data(period_features)
    plot_feature_group(period_features, combined_data_period, "Box Plots for Period Features", 2)

    # Plot for '_time' features
    combined_data_time = create_combined_data(time_features)
    plot_feature_group(time_features, combined_data_time, "Box Plots for Time Features", 3)

    # Tweak layout and save the plot
    plt.tight_layout()
    plt.savefig(f"{path}/5-Box_Plot.png")
    plt.show()

def plot_orignal_pulse_data(input_data):
    """
    Plot the original pulse data for the whole 'data'
    """
    original_pulse = np.array(input_data['data'])  # Ensure it’s a numpy array
    original_pulse_flattened = original_pulse.flatten()  # Flatten to 1D array
    plt.figure(figsize=(12, 6))
    plt.plot(original_pulse_flattened, label='Original Pulse Data')
    plt.title(f"Pulse Data for {input_data['params']['material']}")
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(f"{path}/1_original_pulse.png")
    plt.show()

def plot_master_data(materialname):

    ## Loop through each row in all master data and plot each pulse
    master_data = load_masterdata()
    filtered_data = master_data[master_data['material'] == materialname]
    # Create a figure for all pulse data in one plot
    plt.figure(figsize=(12, 6))

    # Loop through each row in filtered_data and plot each pulse
    for index, row in filtered_data.iterrows():
        profile_data = row['profile']  # Assuming 'profile' contains the pulse data as a list
        trial = row['trial']
        
        # Plot each pulse data
        plt.plot(profile_data, label=f'Trial {trial}')

    # Add plot title and labels
    plt.title(f"Master dataset from PA of {materialname}")
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.ylim(0, 1000)
    # Save and show the combined plot
    plt.savefig(f"{path}/3-Master Data.png")
    plt.show()

def plot_scaled_pulse(input_data):
    plt.figure(figsize=(12, 6))
    plt.plot(input_data[0], label='Scaled Pulse Data', color='red')
    # plt.title(f"Pulse Data for {input_data['params']['material']}")
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.ylim(0, 1000)
    plt.legend()
    plt.savefig( f"{path}/2-scaled_pulse.png")
    plt.show()

def plot_pulse_data(input_data):
    
    data = np.array(input_data['data']).flatten()  # Scaled pulse as a 1D array
    name = input_data['params']['material']

    plt.figure(figsize=(12, 6))
    plt.plot(data, label=name, color='red')
    plt.title(f"{len(data)}")
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig( f"{path}/0-Pulse.png")
    plt.show()

def plot_pulse_calibrate_data(pulses, title):
    """
    Plots multiple pulses on a single graph and saves the plot with a unique filename.
    
    Parameters:
    - pulses (list of lists): A list where each element is a list of amplitude values for a single pulse.
    - title (str): The title of the plot, which is also used in the filename.
    - path (str): Directory path to save the plot. If provided, the plot is saved at this path.
    """
    plt.figure(figsize=(12, 6))
    for i, pulse in enumerate(pulses):
        if len(pulse) > 0:  # Only plot if pulse is not empty
            plt.plot(np.array(pulse).flatten(), label=f"Pulse {i + 1}")
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    if path:
        filename = f"{path}/{title.replace(' ', '_')}_Pulses.png"
        plt.savefig(filename)
    plt.show()
    plt.close()

def plot_boxplot(data, title):
    """
    Plots a boxplot for PA and TP data side by side and saves the plot with a unique filename.
    
    Parameters:
    - data (dict): A dictionary with keys 'PA' and 'TP' containing lists of feature values.
    - title (str): Title of the boxplot, which is also used in the filename.
    - path (str): Directory path to save the plot. If provided, the plot is saved at this path.
    """
    plt.figure(figsize=(8, 6))
    plt.boxplot([data['PA'], data['TP']], labels=['PA', 'TP'])
    plt.title(title)
    plt.ylabel('Feature Value')

    # Create a unique filename using the title
    filename = f"{title.replace(' ', '_')}_Boxplot.png"
    if path:
        os.makedirs(path, exist_ok=True)  # Ensure the directory exists
        filename = os.path.join(path, filename)
    plt.savefig(filename)
    plt.show()
    plt.close()  # Close the figure after saving and displaying

def plot_tsne_with_ref(new_tsne_coordinates, input_data, materialname):
    submap, scaler = load_submap_data()
    material_tsne = submap[submap['material'] == materialname][['TSNE1', 'TSNE2', 'material']]
    sns.set_theme()
    sns.set_style("whitegrid")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure the color palette is valid
    unique_mattype = submap['mattype'].nunique()
    if unique_mattype > len(custom_palette):
        raise ValueError(f"Custom palette has fewer colors ({len(custom_palette)}) than unique mattype categories ({unique_mattype}).")

    # Filter out "mystery" category
    filtered_mattype = submap['mattype'].unique()
    filtered_mattype = [mt for mt in filtered_mattype if mt != "mystery"]
    handles = [mpatches.Patch(color=custom_palette[i], label=mt) for i, mt in enumerate(filtered_mattype)]

    # # KDE plot for density visualization
    sns.kdeplot(
        data=submap[submap['mattype'] != 'mystery'],
        x="TSNE1",
        y="TSNE2",
        hue="mattype",
        palette=custom_palette,
        fill=True,
        alpha=0.5,
        legend=False,
        bw_adjust=0.9,
        ax=ax
    )

    # Add legend to the plot
    legend = ax.legend(
        handles=handles, 
        title="Material Type",
        fontsize='10',
        loc='upper left', 
        bbox_to_anchor=(1, 1),
        borderaxespad=0.
    )
    legend.get_title().set_fontsize('10')

    # Set plot limits
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)

    # Plot the scatter points with Seaborn
    sns.kdeplot(
    data=material_tsne,
    x="TSNE1", y="TSNE2",
    thresh=0.1,          # Set a threshold for the contour density level
    levels=10,           # Number of contour levels
    color="black",        # Set a single color since all points are the same material
    fill=True            # Fill the contours for a more solid look
    )

    # Set axis labels
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")

    ############## PLOT NEW POINT
    new_tsne_point = new_tsne_coordinates
    name = input_data["params"]["name"]

    # Plot and annotate the new point
    ax.scatter(new_tsne_point[0], new_tsne_point[1], color="black", s=50)
    ax.annotate(
        name,
        (new_tsne_point[0], new_tsne_point[1]),
        textcoords="offset points",
        xytext=(5, 5),
        ha='center',
        fontsize=8,
        color='black'
    )
    plt.savefig(f"{path}/6-tsne_plot_with_ref.png")
    plt.show()

def plot_tsne(new_tsne_coordinates, input_data):
    submap, scaler = load_submap_data()
    sns.set_theme()
    sns.set_style("whitegrid")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure the color palette is valid
    unique_mattype = submap['mattype'].nunique()
    if unique_mattype > len(custom_palette):
        raise ValueError(f"Custom palette has fewer colors ({len(custom_palette)}) than unique mattype categories ({unique_mattype}).")

    # Filter out "mystery" category
    filtered_mattype = submap['mattype'].unique()
    filtered_mattype = [mt for mt in filtered_mattype if mt != "mystery"]
    handles = [mpatches.Patch(color=custom_palette[i], label=mt) for i, mt in enumerate(filtered_mattype)]

    # # KDE plot for density visualization
    sns.kdeplot(
        data=submap[submap['mattype'] != 'mystery'],
        x="TSNE1",
        y="TSNE2",
        hue="mattype",
        palette=custom_palette,
        fill=True,
        alpha=0.5,
        legend=False,
        bw_adjust=0.9,
        ax=ax
    )

    # Add legend to the plot
    legend = ax.legend(
        handles=handles, 
        title="Material Type",
        fontsize='10',
        loc='upper left', 
        bbox_to_anchor=(1, 1),
        borderaxespad=0.
    )
    legend.get_title().set_fontsize('10')

    # Set plot limits
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)

    # Set axis labels
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")

    ############## PLOT NEW POINT
    new_tsne_point = new_tsne_coordinates
    name = input_data["params"]["name"]

    # Plot and annotate the new point
    ax.scatter(new_tsne_point[0], new_tsne_point[1], color="black", s=50)
    ax.annotate(
        name,
        (new_tsne_point[0], new_tsne_point[1]),
        textcoords="offset points",
        xytext=(5, 5),
        ha='center',
        fontsize=8,
        color='black'
    )
    plt.savefig( f"{path}/4-tsne_plot.png")
    plt.show()

def plot_chi(new_tsne_coordinates, input_data):
    """
    Plots t-SNE data for the specified materials along with new t-SNE coordinates.

    Args:
        new_tsne_coordinates (tuple): The (x, y) coordinates of the new data point.
        input_data (dict): The input data dictionary containing information about the material.
       
    """
    # Load the submap data
    submap, scaler = load_submap_data()
    # print(submap['material'].unique())
    # Filter the submap to include only the specified materials
    filtered_submap = submap[submap['material'].isin(materialnames)][['TSNE1', 'TSNE2', 'material']]

    # Set up the plot with Seaborn and Matplotlib
    sns.set_theme()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot KDE plots for the selected materials and create legend entries
    for material in materialnames:
        material_data = filtered_submap[filtered_submap['material'] == material]
        sns.kdeplot(
            data=material_data,
            x="TSNE1", y="TSNE2",
            thresh=0.1,          # Set a threshold for contour density
            levels=10,           # Number of contour levels
            label=material,      # Use the material name as the label
            fill=True,           # Fill the contours for a solid look
            alpha=0.5            # Set transparency
        )

    # Plot and annotate the new t-SNE point
    new_tsne_point = new_tsne_coordinates
    name = input_data["params"]["name"]
    # name = "Current Sample"
    ax.scatter(new_tsne_point[0], new_tsne_point[1], color="black", s=50, label=name)
    ax.annotate(
        name,
        (new_tsne_point[0], new_tsne_point[1]),
        textcoords="offset points",
        xytext=(5, 5),
        ha='center',
        fontsize=8,
        color='black'
    )
    # Set plot limits
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)
    # Set axis labels
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")

    # Add legend with all material names from filtered_submap, including the new point
    ax.legend(title="Material", fontsize='10', loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()

    # Save and display the plot
    plt.savefig(f"{path}/7-tsne_plot_selected_materials.png")
    plt.show()

def plot_chi2(new_tsne_coordinates, input_data, materialname):
    # Load and filter the submap data to exclude "mystery" and "ink"
    submap, scaler = load_submap_data()
    submap = submap[(submap['mattype'] != 'mystery') & (submap['mattype'] != 'ink')]

    material_tsne = submap[submap['material'] == materialname][['TSNE1', 'TSNE2', 'material']]

    # Set up the Seaborn theme and plot style
    sns.set_theme()
    sns.set_style("whitegrid")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure the color palette is valid
    unique_mattype = submap['mattype'].nunique()
    if unique_mattype > len(custom_palette):
        raise ValueError(f"Custom palette has fewer colors ({len(custom_palette)}) than unique mattype categories ({unique_mattype}).")

    # Create legend handles for the remaining material types
    filtered_mattype = submap['mattype'].unique()
    handles = [mpatches.Patch(color=custom_palette[i], label=mt) for i, mt in enumerate(filtered_mattype)]

    # KDE plot for density visualization
    sns.kdeplot(
        data=submap,
        x="TSNE1",
        y="TSNE2",
        hue="mattype",
        palette=custom_palette,
        fill=True,
        alpha=0.5,
        legend=False,
        bw_adjust=0.9,
        ax=ax
    )

    # Add legend to the plot
    legend = ax.legend(
        handles=handles, 
        title="Material Type",
        fontsize='10',
        loc='upper left', 
        bbox_to_anchor=(1, 1),
        borderaxespad=0.
    )
    legend.get_title().set_fontsize('10')

    # Set plot limits
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)

    # Plot and annotate the new point
    new_tsne_point = new_tsne_coordinates
    name = input_data["params"]["name"]

    ax.scatter(new_tsne_point[0], new_tsne_point[1], color="black", s=50)
    ax.annotate(
        name,
        (new_tsne_point[0], new_tsne_point[1]),
        textcoords="offset points",
        xytext=(5, 5),
        ha='center',
        fontsize=8,
        color='black'
    )

    # Save and display the plot
    plt.savefig(f"{path}/8-tsne_plot.png")
    plt.show()
