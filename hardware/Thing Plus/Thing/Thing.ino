#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <SparkFun_MicroPressure.h>

// Network credentials
#define WIFI_SSID "XXXX"
#define WIFI_PASS "XXXX"

/// LOCAL
#define WS_ADDR "XXXXX"
#define WS_PORT XXX


// Device and sensor setup
#define BAUD 115200
#define DEVICE_NAME "programmable-air"
#define VBATPIN 9

SparkFun_MicroPressure mpr;
DynamicJsonDocument json(4096 * 4);

WiFiClient wifi;
websockets::WebsocketsClient client;

bool pressure_on = true;
bool isFirstMessage = true;
unsigned long startTime;

void sendJSON(DynamicJsonDocument* json) {
  String message;
  serializeJson(*json, message);
  if (client.send(message)) {
    Serial.println("Message sent successfully:");
  } else {
    Serial.println("Failed to send message:");
  }
  Serial.println(message);
  json->clear();
}

void pressure_read() {
  // uint16_t pressure = mpr.readPressure(TORR);
  float pressurePA = mpr.readPressure(PA);
  uint32_t pressure = static_cast<uint32_t>(pressurePA);
  json["event"] = "read-pressure";
  json["time"] = millis();
  JsonArray data = json.createNestedArray("data");
  data.add(pressure);

  if (isFirstMessage) {
    startTime = millis();
    unsigned long elapsedTime = millis() - startTime;
    float samplingRate = 1.0 / ((float)elapsedTime / 1000.0);
    json["elapsedTime"] = elapsedTime;
    json["samplingRate"] = samplingRate;
    isFirstMessage = false;
  }

  if (json.overflowed()) {
    Serial.println("ERROR: JSON document overflowed!");
  }

  sendJSON(&json);
}

void pump_on(int pumpNumber, int pwm) {
  Serial1.write('1');
  delay(10);
  Serial1.write(pumpNumber);
  delay(10);
  Serial1.write(pwm);
  delay(10);
  Serial1.write('\n');
}

void pump_off(int pumpNumber) {
  Serial1.write('2');
  delay(10);
  Serial1.write(pumpNumber);
  delay(10);
  Serial1.write('\n');
}

void manifest() {
  json["event"] = "manifest";
  json["time"] = millis();
  JsonArray data = json.createNestedArray("data");
  data.add("PRESSURE_ON");
  data.add("PRESSURE_OFF");
  data.add("BATTERY");
  data.add("MANIFEST");
  sendJSON(&json);
}

void apiCall(String prefix, JsonObject obj) {
  char c = '\0';
  Serial.print("API CALL: ");
  Serial.println(prefix);

  if (prefix == "PUMP_ON") {
    pump_on(obj["pumpNumber"], obj["PWM"]);
  } else if (prefix == "PUMP_OFF") {
    pump_off(obj["pumpNumber"]);
  } else if (prefix == "ALL_PUMP_OFF") {
    c = '3';
  } else if (prefix == "PULSE_ON") {
    c = '4';
  } else if (prefix == "PULSE_OFF") {
    c = '5';
  } else if (prefix == "PRESSURE_ON") {
    c = '6';
    pressure_on = true;
    Serial.println("Reading pressure...");
  } else if (prefix == "PRESSURE_OFF") {
    c = '7';
    pressure_on = false;
    Serial.println("Stopped reading pressure...");
  } else if (prefix == "BLOW") {
    c = '8';
  } else if (prefix == "SUCK") {
    c = '9';
  } else if (prefix == "VENT") {
    c = 'a';
  } else if (prefix == "SEAL") {
    c = 'b';
  } else if (prefix == "RELEASE") {
    c = 'c';
  } else if (prefix == "VISCOSENSE") {
    c = 'd';
  } else if (prefix == "SENSE") {
    c = 'e';
  } else if (prefix == "BATTERY") {
    float measuredvbat = analogRead(VBATPIN);
    measuredvbat *= 2;
    measuredvbat *= 3.3;
    measuredvbat /= 1024;
    json["event"] = "battery";
    json["time"] = millis();
    json["data"] = measuredvbat;
    sendJSON(&json);
  } else if (prefix == "MANIFEST") {
    manifest();
  } else {
    Serial.println("COMMAND NOT FOUND");
  }

  if (c != '\0') {
    Serial1.write(c);
    Serial1.write('\n');
  }
}

void apiLoop() {
  static unsigned long lastReadTime = 0;
  const unsigned long readInterval = 10;  // interval in milliseconds

  if (pressure_on) {
    unsigned long currentTime = millis();
    if (currentTime - lastReadTime >= readInterval) {
      pressure_read();
      lastReadTime = currentTime;
    }
  }
}

void listen() {
  if (client.available()) {
    client.poll();
  }
  apiLoop();
}

void setup() {
  Serial.begin(BAUD);
  Serial1.begin(BAUD);

  Wire.begin();
  //Wire.setClock(400000);  // Set I2C clock speed to 400kHz
  if (!mpr.begin()) {
    Serial.println("Cannot connect to MicroPressure sensor.");
    while (1)
      ;
  }

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");

  bool connected = client.connect(WS_ADDR, WS_PORT, "/");
  if (connected) {
    Serial.println("WebSocket Connected!");
    json["name"] = DEVICE_NAME;
    json["event"] = "greeting";
    json["data"] = 0;
    sendJSON(&json);
  } else {
    Serial.println("WebSocket Connection Failed!");
    return;
  }

  client.onMessage([&](websockets::WebsocketsMessage message) {
    Serial.print("Got Message: ");
    String response = message.data();
    Serial.println(response);

    DeserializationError error = deserializeJson(json, response);
    if (error) {
      Serial.print("deserializeJson() failed: ");
      Serial.println(error.c_str());
      return;
    }

    JsonObject obj = json.as<JsonObject>();
    if (obj.containsKey("api")) {
      JsonObject apiObj = obj["api"];
      if (apiObj.containsKey("command")) {
        String command = apiObj["command"];
        JsonObject params = apiObj["params"];
        apiCall(command, params);
      }
    }
    json.clear();
    Serial.println("Message processing complete.");
  });
}

void loop() {
  listen();
}
