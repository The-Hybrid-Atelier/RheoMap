#!/usr/bin/env python3

import libraries.config as config
from libraries.haws import JSONWebSocketClient
import libraries.rheosense as rheosense
import time

# --------- CONFIG --------------
NAME = "rheosense-controller"
jws = JSONWebSocketClient(NAME, config.socket)
idx = 1000

def main():
    """
    Main function to connect to the websocket and handle events.
    """
    jws.connect()

    msg_from_remote = {
      "event": "MODEL_TEST",
      "params": {
          "material": "Water",
          "pulses": 3,
          "name": "Water",
          "color": "red",
          "abbv": "WTR"
      }
    }
    
    jws.send(msg_from_remote)

if __name__ == "__main__":
    main()
