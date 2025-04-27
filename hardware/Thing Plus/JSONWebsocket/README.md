# JSONWebsocket

A WebSocket client for the Hybrid Atelier Websocket Server (HAWS) ("ws://162.243.120.86:443") or ("ws://162.243.120.86:3001") for Adafruit Feather ATWINC1500. 


HWAS has a simple communication protocol.
* It will only accept messages that are JSON strings. This requires using the ArduinoJSON library to parcel messages. 
* Any message you send to the server will be sent to all devices connected to the server, except the original sender. 
* To call an API point in your Feather, the current message format is: `{"api": {"command":"API_POINT", params: {"p1": 5}}}`

# Installation
Clone this repo into your Arduino libraries folder 
```
cd ~/Documents/Arduino/libraries 
git clone https://github.com/The-Hybrid-Atelier/JSONWebsocket.git
```
Create an arduino_secrets.h file in the `JSONWebsocket/src` directory. Use the arduino_secrets_template.h as a base. Update corresponding values. 

Open one of the examples (File > Examples > JSONWebsocket to get a quick start. 

# Configuring your device

1. In your Arduino sketch, specify a unique device name: `#define DEVICE_NAME "your-device-name"`
2. If you are not using the HWAS server, update the server information in the arduino_secrets.h file. Note that you do not need to make a separate arduino_secrets.h file for each ino sketch. These settings are delegated to the library at `~/Documents/Arduino/libraries/JSONWebsocket/src`.

Two function handlers are required to handle messages from the server:
```
/* Function Prototypes */
void apiCall(WebSocketClient client, String prefix, JsonObject obj);
void apiLoop(); 

/* WS Client Init */
JSONWebsocket ws (DEVICE_NAME, Serial, BAUD, apiCall, apiLoop);
```

```
void apiCall(WebSocketClient client, String prefix, JsonObject obj){
  Serial.println(prefix);
  /* Logic of what to do on server commands */
  if (prefix == "API_POINT")         { some_arduino_function(obj[String("p1")]);}
  //... more API points
  else { Serial.write("COMMAND NOT FOUND\n");}
}
```
and 
```
void apiLoop() {
/* Anything you'd like to have run after each message poll.
}
```
In the Arduino's loop function, call:
```
ws.listen();
```
This function runs the client code the sees if a new message is available. 

Upload your code. Run the `SERVER_STATE` command on the [HAWS daemon](http://humrattlepurr.cearto.com/iot/server) to see if your device successfully connected to the HAWS. AdafruitFeather typically needs a Reset button double click to enter upload mode.

# Handling a message
Consider a `GPIO` API point that turns a GPIO pin HIGH(1) or LOW(0).
The message from other devices would look like:
`{"api": {"command":"GPIO", params: {"pin": 4, "value": 1}}}`

To enable handling this message within our Feather, in our `apiCall` function, we would add a condition for `GPIO` as follows:
```
else if (prefix == "GPIO"){ gpio(obj[String("pin")], obj[String("value")]);}
```
and write a handler for the API point:
```
void gpio(int pin, int value){
  digitalWrite(pin, value);
}
```


# Sending a message
We typically keep one buffer (`json`) for our messages.
```
/* Websocket Server Buffer and Object */
DynamicJsonDocument json(4096*4);
```
When we want to send a message, we clear the buffer and write in the data as follows:
```
/* Server Logic */
void manifest(WebSocketClient client){  
  json["event"] = "manifest";
  json["time"] = millis();

  JsonArray data = json.createNestedArray("data");
  data.add("LED_ON");
  data.add("LED_OFF");
  data.add("BATTERY");
  data.add("MANIFEST");
  
  ws.send(&json); /* .send will also clean the json buffer */
}
```
We can send the manifest message (`{"event": "manifest", "time": 10390193, "data":["LED_ON", ...]}`) by calling:
`manifest(ws);`

# Testing

* The [HAWS web client](http://humrattlepurr.cearto.com/iot/prototype) allows you to send JSON messages to HAWS clients.
* The [HAWS daemon](http://humrattlepurr.cearto.com/iot/server) allows you to send a special "SERVER_STATE" message that logs all devices currently connected to the server. 


