#include <SoftwareSerial.h>
#include "programmable_air.h"
#include <Adafruit_NeoPixel.h>
#include "behaviors.h"

Adafruit_NeoPixel px = Adafruit_NeoPixel(3, neopixelPin, NEO_GRB + NEO_KHZ800);
#define DEBUG 1
#define BAUDRATE 115200

char c;
int buff;
bool pressure = false;
bool pulse = false;
int pumpNumber;
int pwm;
bool first = true;

// Programmable Air Constant //
int state = UN_KNOWN;
int threshold = 8;
int switch_ = 9;

void setup() {
  initializePins();
  pinMode(switch_, INPUT_PULLUP);
  px.begin();
  Serial.begin(BAUDRATE);
  Serial.println("Nano initialized. Waiting for commands...");
}

/// @brief Function Code Handler ////
void check_end(char c) {
  while (c != '\n') {
    if (Serial.available() > 0) {
      c = Serial.read();
    }
  }
}

void pumpOn(char c) {
  while (c != '\n') {
    if (Serial.available() > 0) {
      buff = Serial.read();
      if (buff == '\n') {
        c = buff;
        break;
      } else {
        c = 'x';
        if (first) {
          pumpNumber = buff;
          first = false;
        } else {
          pwm = buff;
          first = true;
        }
      }
    }
  }
  Serial.print("Turning on pump ");
  Serial.print(pumpNumber);
  Serial.print(" with PWM ");
  Serial.println(pwm);
  switchOnPump(pumpNumber, pwm);
}

void pumpOff(char c) {
  while (c != '\n') {
    if (Serial.available() > 0) {
      buff = Serial.read();
      if (buff == '\n') {
        c = buff;
        break;
      } else {
        pumpNumber = buff;
      }
    }
  }
  Serial.print("Turning off pump ");
  Serial.println(pumpNumber);
  switchOffPump(pumpNumber);
}

void seal() {
  Serial.println("Sealing...");
  closeAllValves();
}

void check() {
  Serial.println("Checking...");
  setAllValves(OPEN);
}

/// @brief Main code handler ////
void handler() {
  // THE FIRST CHARACTER OF A COMMAND LOGIC  
  c = Serial.read();
  Serial.print("Received command: ");
  Serial.println(c);

  if (c == '1') {
    pumpOn(c);
  } else if (c == '2') {
    pumpOff(c);
  } else if (c == '3') {
    Serial.println("Switching off all pumps");
    switchOffPumps();
    check_end(c);
  } else if (c == '4') {
    pulse = true;
    Serial.println("Pulsing on");
    check_end(c);
  } else if (c == '5') {
    pulse = false;
    Serial.println("Pulsing off");
    check_end(c);
  } else if (c == '6') {
    Serial.println("PRESSURE ON - RheoPulse Routine");
    switchOffPumps();
    rheosense_routine();
    check_end(c);
  } else if (c == '7') {
    Serial.println("PRESSURE OFF - Switching off pumps");
    switchOffPumps();
    check_end(c);
  } else if (c == '8') {
    Serial.println("Clearing out");
    clear_out();
    check_end(c);
  } else if (c == '9') {
    Serial.println("Sucking");
    suck();
    check_end(c);
  } else if (c == 'a') {
    Serial.println("Venting");
    vent();
    check_end(c);
  } else if (c == 'b') {
    seal();
    check_end(c);
  } else if (c == 'c') {
    check();
    check_end(c);
  } else if (c == 'd') {
    Serial.println("Running rheosense routine");
    rheosense_routine();
    check_end(c);
  } else if (c == 'e') {
    Serial.println("Received 'e' message from Thing Plus");
    switchOffPumps();
    check_end(c);
  } else {
    Serial.println("Unknown command received");
  }
}

/// @brief Physical Button Handler ////

bool blueButtonPressed = false;  // To track if the blue button has been pressed
bool blueButtonToggled = false;  // To track the state of the blue button routine
unsigned long lastRoutineTime = 0;  // To store the last time the rheosense_routine() was called
const unsigned long routineInterval = 3000;  // 3 seconds in milliseconds

unsigned long lastBluePressTime = 0;  // To store the last time the blue button was pressed
const unsigned long doubleClickThreshold = 500;  // Max interval for a double click in milliseconds

void physical_button() {
  unsigned long currentTime = millis();
  
  // If blue button is pressed
  if (readBtn(BLUE) || !digitalRead(switch_)) {
    // If it's a double-click
    if (currentTime - lastBluePressTime < doubleClickThreshold) {
      blueButtonToggled = !blueButtonToggled;  // Toggle the blue button state
      blueButtonPressed = false;  // Reset single press tracking
      lastRoutineTime = millis();  // Reset the timer
      if (blueButtonToggled) {
        rheosense_routine();  // Call the routine immediately on toggle on
      }
    } else {
      // Record the time of this single press
      lastBluePressTime = currentTime;
    }
    blueButtonPressed = true;  // Track that the blue button was pressed
  }
  
  // If blue button is released, reset the pressed state
  if (!readBtn(BLUE) && digitalRead(switch_)) {
    blueButtonPressed = false;
  }
  
  // If red button is pressed, start the blow function
  if (readBtn(RED)) {
    switchOnPump(2, 100);
    switchOffPump(1);
    blow();
  } else {
    // Stop the blow function as soon as the red button is released
    switchOffPumps();
  }

  // If blue button toggled is true and 3 seconds have passed since the last routine call
  if (blueButtonToggled && (millis() - lastRoutineTime >= routineInterval)) {
    lastRoutineTime = millis();  // Reset the timer
    rheosense_routine();  // Call the routine
  }

  // Existing code for venting
  if (!readBtn(BLUE) && !readBtn(RED) && digitalRead(switch_) && state != VENTING) {
    switchOffPumps();
  }
}


////////////////// Main loop

void loop() {
  if (Serial.available() > 0) handler();
  if (pulse) pulsing();
  delay(50);
  physical_button();
}
