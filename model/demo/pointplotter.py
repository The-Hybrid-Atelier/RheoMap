# pylint: disable=all

import libraries.config as config
from libraries.haws import JSONWebSocketClient
import libraries.rheosense as rheosense
import time
import random
import argparse

# Argument parser function
def extract_args():
    parser = argparse.ArgumentParser(description="Rheomap Routine Demo")
    parser.add_argument('-x', '--x', type=float, help="X-coordinate of the point (optional)")
    parser.add_argument('-y', '--y', type=float, help="Y-coordinate of the point (optional)")
    
    args = parser.parse_args()
    print("Running RheoMap Plot Point Simulator")
    
    return args

# Function to send a point via WebSocket
def simulate_point(jws, x, y):
    jws.send({"event": "plotpoint", "pt": {"x": x, "y": y}})

def main():
    args = extract_args()
    
    # Initialize WebSocket
    NAME = "rheosense-simulator"
    jws = JSONWebSocketClient(NAME, config.socket)
    jws.connect()
    
    # while True:
        # Generate random point if x or y are not provided
    x = args.x if args.x is not None else random.uniform(-100, 100)
    y = args.y if args.y is not None else random.uniform(-100, 100)
    
    # Simulate and send point
    simulate_point(jws, x, y)
    print(f"Sent point: x={x}, y={y}")
        
    # time.sleep(0.5)  # Adjust delay between points as needed

if __name__ == "__main__":
    main()
