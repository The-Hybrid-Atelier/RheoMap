
# pylint: disable=all

import libraries.config as config
from libraries.haws import JSONWebSocketClient
import libraries.rheosense as rheosense
import time


# python mapviewer.py -n "Water"

import argparse
def extract_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Rheomap Routine Demo")

    # Add the '-n' argument for name
    parser.add_argument('-n', '--neighborhood', type=str, required=True, help="Name of neighborhood")

    # Parse the command-line arguments
    args = parser.parse_args()

    print("Running RheoMap Routine Demo on sample: ", args.neighborhood)
    return (args.neighborhood)



def main():
    NAME = "map-controller"
    jws = JSONWebSocketClient(NAME, config.socket)
    neighborhood = extract_args()

    # -------- OPEN CONNECTION ------
    jws.connect()
    
    # # Run the routine
    jws.send({"event": "toggle-neighborhood", "data": neighborhood})

    # # Wait for the routine to finish

if __name__ == "__main__":
    main()
