# WEBSOCKET URL IS FOUND IN LIBRARIES/CONFIG.PY
import libraries.config as config
from libraries.haws import JSONWebSocketClient
import libraries.rheosense as rheosense
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

# Initialize data storage
x_data, y_data = [], []

# Apply Seaborn theme for styling
sns.set_theme()

# Initialize the plot
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter([], [])  # Initialize with empty data
ax.set_xlim(-100, 100)  # Set x-axis limits as required
ax.set_ylim(-100, 100)  # Set y-axis limits as required

# Function to handle incoming data
def dataread_handler(jws, msg, obj):
    # Extract x and y from the received message
    point = msg.get("pt", {"x": 0, "y": 0})
    x, y = point["x"], point["y"]
    print("Point received:", (x, y))

    # Append the new point to the data lists
    x_data.append(x)
    y_data.append(y)

def main():
    NAME = "rheomap-tester"
    jws = JSONWebSocketClient(NAME, config.socket)

    # -------- OPEN CONNECTION ------
    jws.connect()
    jws.subscribe("rheosense-simulator", "plotpoint")
    jws.on("rheosense-simulator", "plotpoint", dataread_handler)

    while True:
        # Listen for WebSocket messages
        jws.listen()

        # Update the plot with new points if available
        if x_data and y_data:
            scatter.set_offsets(np.c_[x_data, y_data])  # Update all points

        plt.pause(0.5)  # Small pause for both plotting and WebSocket listening

if __name__ == "__main__":
    main()
