#ifndef JSON_WS
#define JSON_WS

// #include <ArduinoHttpClient.h>
#include <ArduinoWebsockets.h>
//#include <WiFi101.h>
#include <WiFi.h>  // Update to WiFi library compatible with ESP32
#include <string.h>
#include <ArduinoJson.h>
#include "arduino_secrets.h"

// typedef void (*functype)(WebsocketsClient client, String prefix, JsonObject obj);
// typedef void (*functype2)(void);
using functype = void (*)(websockets::WebsocketsClient client, String prefix, JsonObject obj);
using functype2 = void (*)(void);

class JSONWebsocket{
	public:
		JSONWebsocket (char* name, Stream & port, int baud, functype api, functype2 loop) : port_ (port), name_(name), baud_(baud), api_(api), loop_(loop){ }
		IPAddress ip;
		void init();
		void enable_api();
		void listen();
		void greet();
		void battery();
		void send(DynamicJsonDocument *json);
	private: 
		Stream &port_;
		char* name_;
		int baud_;
		functype api_;
		functype2 loop_;
		int status;
		int count;
		
}; 

#endif

