#include "DHT.h"

// --- Pin Definitions (Same Config) ---
#define DHTPIN 6          
#define DHTTYPE DHT11     
#define FLAME_PIN 7       // Digital: LOW = Fire detected
#define SMOKE_PIN A0      // Analog: Higher = Smoke detected
#define PUMP_PIN 8        // Relay: HIGH = Pump ON

// Motor Pins (L298N)
const int IN1 = 2; const int IN2 = 3; 
const int IN3 = 4; const int IN4 = 5; 

// --- Thresholds ---
const int smokeThreshold = 250;    // Lowered for faster detection
const float tempThreshold = 45.0;  

DHT dht(DHTPIN, DHTTYPE);

unsigned long lastWanderTime = 0;
int wanderDirection = 0; 

void setup() {
  dht.begin();
  
  pinMode(FLAME_PIN, INPUT);
  pinMode(PUMP_PIN, OUTPUT);
  
  // REVERSE LOGIC: Ensure pump is OFF at startup
  digitalWrite(PUMP_PIN, LOW); 
  
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  
  randomSeed(analogRead(A5)); // Seed for random patrol
}

void loop() {
  int flameStatus = digitalRead(FLAME_PIN); 
  int smokeLevel = analogRead(SMOKE_PIN);
  float currentTemp = dht.readTemperature();

  if (isnan(currentTemp)) currentTemp = 0; 

  // --- TARGET DETECTED (Priority Attack Mode) ---
  if (flameStatus == LOW || smokeLevel > smokeThreshold || currentTemp > tempThreshold) {
    
    // 1. Move straight into the detected path
    moveForward();
    
    // 2. Instant Pump ON (Reverse Logic: HIGH = ON)
    digitalWrite(PUMP_PIN, HIGH); 
    
    // 3. Stay in "Attack Mode" as long as the sensor is triggered
    // This forces the relay to follow the sensor signal exactly
    while(digitalRead(FLAME_PIN) == LOW || analogRead(SMOKE_PIN) > smokeThreshold) {
      moveForward(); 
      delay(10); 
    }
    
    // 4. Sensors clear: Stop pump and redirect
    digitalWrite(PUMP_PIN, LOW); 
    reverse();
    delay(1000);
    
  } else {
    // --- RANDOM PATROL MODE ---
    digitalWrite(PUMP_PIN, LOW); 
    handleWandering();
  }
}

// --- Navigation Functions ---

void handleWandering() {
  if (millis() - lastWanderTime > 2000) {
    wanderDirection = random(0, 3); 
    lastWanderTime = millis();
  }

  if (wanderDirection == 0) moveForward();
  else if (wanderDirection == 1) turnLeft();
  else if (wanderDirection == 2) turnRight();
}

void moveForward() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void turnLeft() {
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void turnRight() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
}

void reverse() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}

void stopRobot() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}