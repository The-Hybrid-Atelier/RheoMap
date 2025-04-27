import libraries.config as config
from libraries.haws import *

# --------- CONFIG --------------
NAME = "rheosense-tester"
jws = JSONWebSocketClient(NAME, config.socket)

# -------- OPEN CONNECTION ------
jws.connect()

# SEND A MESSAGE

#jws.send_api("RECORD_START")
#jws.send_api("SENSE")
#jws.send_api("CHARLIE")
jws.send_api("RECORD_END_AND_MODEL")

# jws.send(config.read_pressure)
# jws.send(config.read_pressure)
# jws.send(config.read_pressure)
# jws.send(config.read_pressure)
# jws.send(config.read_pressure)
#jws.send_api("RECORD_END", {"material": "honeytest"})



# Converts to valid JSON object
# jws.send({"event":"matread", "data": ["Hello!"]}) #equivalent to send_event
# jws.send_event("matread", config.honeytest)
# jws.send_event("matmodel", config.test2)
# time.sleep(1.0)
jws.close()
