# pylint: disable=all

import libraries.config as config
from libraries.haws import JSONWebSocketClient
import libraries.rheosense as rheosense
import time


# python rheodemo.py -n "Water" -t 1
import argparse
def extract_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Rheomap Routine Demo")

    # Add the '-n' argument for name
    parser.add_argument('-n', '--name', type=str, required=True, help="Name of sample")

    # Add the '-t' argument for an integer value
    parser.add_argument('-t', '--times', type=int, required=True, help="Number of times to repeat")

    # Parse the command-line arguments
    args = parser.parse_args()

    print("Running RheoMap Routine Demo on sample: ", args.name, " for ", args.times, " times")
    return (args.name, args.times)

def routine(jws, name, times):
    START_PULSE_GAP = 0.5  # in seconds
    PULSE_TIME = 2  # in seconds
    END_GAP = 0.1  # in seconds

    print(f"SENSING {name} x {times}")

    for pulse_id in range(times):
        print("PULSE", pulse_id)
        # start_pulse_time = pulse_id * (PULSE_TIME + START_PULSE_GAP + END_GAP)
        # start_pulse_time = (PULSE_TIME + START_PULSE_GAP + END_GAP)
        # time.sleep(start_pulse_time)  # Delay before starting the recording

        # Start the recording
        jws.jsend({"api": {"command": "RECORD_START"}})
        # print(f"timeline: start {start_pulse_time}")

        # Wait and then send the sense command
        time.sleep(START_PULSE_GAP)
        jws.jsend({"api": {"command": "SENSE", "params": {"material": "test"}}})
        # print(f"timeline: sense {start_pulse_time + START_PULSE_GAP}")

        # end_pulse_time = start_pulse_time + START_PULSE_GAP + PULSE_TIME
        end_pulse_time = PULSE_TIME + END_GAP

        if pulse_id != times - 1:
            # End the recording after this pulse
            time.sleep(end_pulse_time)
            jws.jsend({
                "api": {
                    "command": "RECORD_END",
                    "params": {
                        "material": name,
                        "pulses": pulse_id,
                        "name": name,
                    }
                }
            })
            print(f"timeline: record end {end_pulse_time}")
        else:
            # End the recording and trigger the model after the last pulse
            time.sleep(end_pulse_time)
            jws.jsend({
                "api": {
                    "command": "RECORD_END_AND_MODEL",
                    "params": {
                        "material": name,
                        "pulses": pulse_id,
                        "name": name
                    }
                }
            })
            print(f"timeline: record end and model {end_pulse_time}")

def main():
    NAME = "rheosense"
    jws = JSONWebSocketClient(NAME, config.socket)
    name, times = extract_args()

    # -------- OPEN CONNECTION ------
    jws.connect()
    
    # # Run the routine
    routine(jws, name, times)

    # # Wait for the routine to finish

if __name__ == "__main__":
    main()

