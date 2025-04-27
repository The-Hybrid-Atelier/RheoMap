#include "JSONWebsocket.h"

char ssid[] = WIFI_SSID;     // Your network SSID (name)
char pass[] = WIFI_PASS;     // Your network password  
char serverAddress[] = WS_ADDR;  // Server address
int port = WS_PORT;

WiFiClient wifi;
websockets::WebsocketsClient client_;

DynamicJsonDocument server_json(1024);
DynamicJsonDocument feather_json(1024);

void JSONWebsocket::send(DynamicJsonDocument* json) {
    String message;
    serializeJson(*json, message);
    if (client_.send(message)) {
        Serial.println("Message sent successfully:");
    } else {
        Serial.println("Failed to send message:");
    }
    Serial.println(message);
    json->clear();
}

void JSONWebsocket::init() {
    Serial.println("Connecting to WiFi...");
    WiFi.begin(ssid, pass);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println("");
    Serial.println("WiFi connected.");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());

    port_.print("\tAttempting connection to ");
    port_.print(serverAddress);
    port_.print(" from ");
    port_.println(ip);

    bool connected = client_.connect(serverAddress, port, "/");
    if (connected) {
        Serial.println("WebSocket Connected!");
        greet();
    } else {
        Serial.println("WebSocket Connection Failed!");
        return; // Early exit if connection fails
    }

    delay(500);
    
    // Message processing
    client_.onMessage([&](websockets::WebsocketsMessage message) {
        Serial.print("Got Message: ");
        String response = message.data();
        port_.println(response);

        deserializeJson(server_json, response);
        JsonObject obj = server_json.as<JsonObject>();

        if (obj.containsKey("api")) {
            obj = obj["api"];
            if (obj.containsKey("command")) {
                String command = obj["command"];
                obj = obj["params"];
                port_.print("API CALL: ");
                port_.println(command);
                api_(client_, command, obj);
            }
        }
        server_json.clear();
        port_.println("Message processing complete.");
    });
}

void JSONWebsocket::greet() {
    feather_json["name"] = name_;
    feather_json["event"] = "greeting";
    feather_json["data"] = count++;
    send(&feather_json);
}

void JSONWebsocket::battery() {
    float measuredvbat = analogRead(VBATPIN);
    measuredvbat *= 2;    // we divided by 2, so multiply back
    measuredvbat *= 3.3;  // Multiply by 3.3V, our reference voltage
    measuredvbat /= 1024; // Convert to voltage
    port_.print("VBat: "); 
    port_.println(measuredvbat);

    feather_json["event"] = "battery";
    feather_json["time"] = millis();
    feather_json["data"] = measuredvbat;

    send(&feather_json);
}

void JSONWebsocket::listen() {
    if (client_.available()) {
        client_.poll();
    }
    loop_();
}
