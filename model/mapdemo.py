import libraries.config as config
from libraries.haws import JSONWebSocketClient
import libraries.rheosense as rheosense
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from tsne_lib.tsne_and_scale import *
# Initialize data storage
x_data, y_data = [], []

def plot_setup():
    submap, scaler = load_submap_data()  # Load once globally
    # Apply Seaborn theme for styling
    sns.set_theme()
    sns.set_style("whitegrid")  # Clean grid style for publication
    # sns.set_context("paper", font_scale=2.8, rc={"lines.linewidth": 1.5})  # Adjust font size and line width for publication
        
    # Initialize the plot
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    # Define custom color palette
    custom_palette = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#6A5ACD', '#FF1493']  # Tomato, SteelBlue, LimeGreen, Gold, SlateBlue, DeepPink

    # Validate palette length
    unique_mattype = submap['mattype'].nunique()
    if unique_mattype > len(custom_palette):
        raise ValueError(f"Custom palette has fewer colors ({len(custom_palette)}) than unique mattype categories ({unique_mattype}).")

    # Filter out "mystery" and get unique mattype values
    filtered_mattype = submap['mattype'].unique()
    filtered_mattype = [mt for mt in filtered_mattype if mt != "mystery"]
    handles = [mpatches.Patch(color=custom_palette[i], label = mt) for i, mt in enumerate(filtered_mattype)]
    # KDE plot for density visualization
    sns.kdeplot(
        data=submap[submap['mattype'] != 'mystery'],  # Filter data directly in plot
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

    # Add the legend to the plot
    legend = ax.legend(
        handles=handles, 
        title="Material Type",  # Legend title
        fontsize='10',  # Font size for the legend labels
        loc='upper left', 
        bbox_to_anchor=(1, 1), 
        borderaxespad=0.
    )
    # Set a smaller font size for the legend title
    legend.get_title().set_fontsize('10')  # Adjust the font size of the legend title here

    scatter = ax.scatter([], [])  # Initialize with empty data
    ax.set_xlim(-120, 120)  # Set x-axis limits as required
    ax.set_ylim(-120, 120)  # Set y-axis limits as required

def plot(name, input_data):
    plt.plot(input_data)
    plt.title(f"Material: {name} n={len(input_data)}")
    plt.show()

def dataread_handler(jws, msg, obj):
    # -- Grab the last pulse
    params = msg.get("params", {})
    name = params.get('name', "No name provided")
    pressure_data = msg.get("data", [])

    ############
    print("Material:", name)
    # plot(name, pressure_data)
    if len(pressure_data) == 0:
        print("No pulse received")
        return  # Use return instead of quit to continue listening
    ###############
    new_tsne_point = process_and_plot(msg)
    print("New t-SNE coordinates for the projected data point:", new_tsne_point)
    # Append the new point to the data lists
    x_data.append(new_tsne_point[0])
    y_data.append(new_tsne_point[1])
    
    # Add annotation with name for the new point
    ax.annotate(
        name,  # The name to display
        (new_tsne_point[0], new_tsne_point[1]),  # The (x, y) coordinates
        textcoords="offset points",  # Position the text slightly offset from the point
        xytext=(5, 5),  # Offset values (adjust as needed for visibility)
        ha='center',  # Horizontal alignment
        fontsize=8,  # Font size for the label
        color='black'  # Color for the label text
    )

def toggle_handler(jws, msg, obj):
    print(msg);
    
def main():
    NAME = "rheomap"
    jws = JSONWebSocketClient(NAME, config.socket)

    # -------- OPEN CONNECTION ------
    jws.connect()
    jws.subscribe("rheosense", "matsense")
    jws.on("rheosense", "matsense", dataread_handler) 

    jws.subscribe("mapcontroller", "toggle-neighborhood");
    jws.on("mapcontroller", "toggle-neighborhood", toggle_handler);
    
    while True:
        # Listen for WebSocket messages
        jws.listen()

        # Update the plot with new points if available
        if x_data and y_data:
            scatter.set_offsets(np.c_[x_data, y_data])  # Update all points

        plt.pause(0.5)  # Small pause for both plotting and WebSocket listening

if __name__ == "__main__":
    plot_setup()
    main()
