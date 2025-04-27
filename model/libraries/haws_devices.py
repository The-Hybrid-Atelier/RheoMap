
from libraries.haws import *

class HAWSDevice:
  def __init__(self, jwsptr):
    self.jws = jwsptr
    self.handlers = []
  def api(self, command, params={}): 
    return {"api": {"command": command, "params": params}}
  def send(self, message):
    new_state = json.dumps(message, sort_keys=True)
    if new_state == self.state:
      return
    else:
      self.state = new_state
      self.jws.send(message)
  # Threaded helper functions
  def connect(self):
    self.jws.connect()
    for a, b, c in self.handlers:
      print("\t", "Binding handlers: ", self.jws.name, a, b)
      self.jws.on(a, b, c)
    if hasattr(self.jws, "start"):
      self.jws.start()
  def listen(self):
    return self.jws.listen()
  def shutdown(self):
    self.active = False
    self.send_off()
    if hasattr(self.jws, "shutdown"):
      self.jws.shutdown()
  def set_limit(self, limit):
    if hasattr(self.jws, "set_limit"):
      self.jws.set_limit(limit)
  def join(self):
    if hasattr(self.jws, "join"):
      self.jws.join()


class HAWSMicrophoneAPI(HAWSDevice):
  def __init__(self, jwsptr):
    super().__init__(jwsptr)
  def print_handler(jws, msg, obj):
    print(msg)
  def set_data_handler(self, data_handler):
    self.handlers.append(("audio-streamer", "mic-read", data_handler))
  def send_on(self, rate=0):
    self.jws.send(self.api("MIC_ON", {"rate": rate}))
  def send_off(self):
    self.jws.send(self.api("MIC_OFF"))


class HAWSOhmmeter(HAWSDevice):
  def __init__(self, jwsptr):
    super().__init__(jwsptr)
  def print_handler(jws, msg, obj):
    print(msg)
  def set_data_handler(self, data_handler):
    self.handlers.append(("ohmmeter", "meter-read", data_handler))
  def led(self, state=0):
    if state == 0:
      self.jws.send(self.api("LED_OFF"))
    else:
      self.jws.send(self.api("LED_ON"))
  def measure(self, property, pin="A0"):
    if property == "resistance":
      self.jws.send(self.api("MEASURE", {"property": "resistance"}))
    elif property == "voltage_10bit":
      self.analogRead(pin)

  def analogRead(self, pin):
    try:
      if isinstance(pin, str) and pin[0] == "A":
        pin = int(pin[1]) + 14
      if not(pin in range(14, 20)):
        raise TypeError("Expecting 14-19 or A0-A5 (Adafruit MO analog pins):", pin)
      self.jws.send(self.api("ANALOG_READ", {"pin": pin}))
    except:
      print("Ignoring command for analogRead", pin)
      raise
      
  def send_off(self):
    self.led(0)
  
    
class HAWSNeopixel(HAWSDevice):
  RED = {"red": 255, "green": 0, "blue": 0}
  ORANGE = {"red": 255, "green": 165, "blue": 0}
  YELLOW = {"red": 255, "green": 255, "blue": 0}
  GREEN = {"red": 0, "green": 255, "blue": 0}
  BLUE = {"red": 0, "green": 0, "blue": 255}
  PURPLE = {"red": 255, "green": 0, "blue": 255}
  WHITE = {"red": 255, "green": 255, "blue": 255}
  RAINBOW = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
  
  def __init__(self, jwsptr):
    super().__init__(jwsptr)
    self.state = ""
  def device_brightness(self, brightness):
    self.send(self.api("LED_BRIGHTNESS", {"brightness": brightness}))
  def on(self):
    self.send(self.api("LED_ON"))
  def off(self):
    self.send(self.api("LED_OFF"))
  def brightness(self, b):
    return {"red": b, "green": b, "blue": b}
  def color_obj(self, color, instantaneous = False):
    command = "LED_COLOR"
    if instantaneous:
      command = "LED_COLOR_INST"
    self.send(self.api(command, color))
  def color(self, r, g, b, instantaneous = False):
    command = "LED_COLOR"
    if instantaneous:
      command = "LED_COLOR_INST"
    self.send(self.api(command, {"red": r, "green": g, "blue": b}))
  def fade(self, delay_time):
    self.send(self.api("LED_FADE", {"delayTime": delay_time}))
  def blink(self, num_of_blinks, delay_time):
    self.send(self.api("LED_BLINK", {"numOfBlinks": num_of_blinks, "delayTime": delay_time}))
  def rgb_wheel(self):
    self.send(self.api("RGB_WHEEL"))

# Buzzer
# https://github.com/The-Hybrid-Atelier/JSONWebsocket/blob/main/examples/RFIDBuzzerHAWS/RFIDBuzzerHAWS.ino
class HAWSBuzzer(HAWSDevice):
  def __init__(self, jwsptr):
    super().__init__(jwsptr)
  def buzz(self, frequency, duration):
    self.jws.send(self.api("buzz", {"frequency": frequency, "duration": duration}))

# RFID
# https://github.com/The-Hybrid-Atelier/JSONWebsocket/tree/main/examples/RFIDHAWS
# Data handler only
# the rfid-read message has 3 keys: event, time, uid
# Data handler only -- not implemented (check commit comment)
class HAWSRFID(HAWSDevice):
  def print_handler(jws, msg, obj):
    print(msg)
  def __init__(self, jwsptr):
    super().__init__(jwsptr)
  def set_data_handler(self, data_handler):
    self.jws.on("rfid-1", "rfid-read", data_handler)
