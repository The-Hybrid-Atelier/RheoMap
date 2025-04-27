#!/usr/bin/env python3

import libraries.config as config
from libraries.haws import *
import libraries.rheosense as rheosense


# --------- CONFIG --------------
NAME = "rheosense-model"
jws = JSONWebSocketClient(NAME, config.socket)
idx = 1000

# Event Handlers
def dataread_handler(jws, msg, obj):
  global idx
  print(msg)
  data = msg["data"]
  name = msg["params"].get("material", "mystery")
  mid = msg["time"]

  try:
    entry = rheosense.sense(data, name, idx)
    idx+=1
    msg = jws.send_event(config.MODELING_EVENT, entry)
    print(config.MODELING_EVENT)
    print(msg)
  except Exception as e:
    print(e)
    msg = jws.send_event("failed-model", entry)
    print(msg)

  


# -------- OPEN CONNECTION ------
jws.connect()

# ------- CONFIGURE SERVICES ------
# print("programmable-air", config.DATAREAD_EVENT)
# jws.subscribe("rheosense-controller", config.DATAREAD_EVENT)
jws.on("rheosense-controller", config.DATAREAD_EVENT, dataread_handler) # listen to all events (*) sent by haws-server; pass info to handler


# -------- LOGIC ----------

# --- LOAD MODELS ----
rheosense.load_models(config.DATA_DIRECTORY)

# SEND A MESSAGE
# message = {"event": "hello-world"}
# Converts to valid JSON object
# jws.send(message)



# LISTEN
# try: 
while True:
  jws.listen()
  time.sleep(1.0)
# except Exception as e:
#   print(e)
#   print("Closing socket.")
#   jws.close()