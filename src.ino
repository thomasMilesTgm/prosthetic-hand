/****************************************************************************
* Sensor API Copyright (C) 2011 - 2014 Bosch Sensortec GmbH. 
* Nine axis shield API copyright Arguino Org:
* https://github.com/arduino-org/arduino-library-nine-axes-motion
*/

#include "NineAxesMotion.h"        
#include <Wire.h>
#define N_LABELS 5
#define NO_LABEL -1

NineAxesMotion mySensor;

const int startPin = 8;
const int changePin = 10;
float label = NO_LABEL;


void setup() 
{

  //Peripheral Initialization
  I2C.begin();                    //Initialize I2C communication to the let the library communicate with the sensor.
  //Sensor Initialization
  mySensor.initSensor();          //The I2C Address can be changed here inside this function in the library
  mySensor.setOperationMode(OPERATION_MODE_NDOF);   //Can be configured to other operation modes as desired
  mySensor.setUpdateMode(MANUAL);
  Serial.begin(115200);             // Initialize serial communication
}

void loop() 
{
  /*
   * When an entire packet is complete will send data packet containing the following information to host.
   * 
   * t    (float - timestamp)
   * l    (float - data label)
   * x    (float - x acceleration)
   * y    (float - y acceleration)
   * z    (float - z acceleration)
   * c0   (float - EMG channel 0)
   * c1   (float - EMG channel 1)
   * c2   (float - EMG channel 2)
   * c3   (float - EMG channel 3)
   */


  
  float packet[9];        // packet containing 8 32 bit floats as above

  packet[0] = millis();   // timestamp
  packet[1] = label; // label for packet TODO, make dynamic

  mySensor.updateAccel();

  // generate the xyz float
  packet[2] = mySensor.readAccelerometer(X_AXIS);
  packet[3] = mySensor.readAccelerometer(Y_AXIS);
  packet[4] = mySensor.readAccelerometer(Z_AXIS);
  packet[5] = analogRead(A0); // c0 
  packet[6] = analogRead(A1); // c1
  packet[7] = analogRead(A2); // c2
  packet[8] = analogRead(A3); // c3

    Serial.print("t!");
    Serial.print(packet[0]);
    Serial.print(" l!");
    Serial.print(packet[1]);
    Serial.print(" x!");
    Serial.print(packet[2]);
    Serial.print(" y!");
    Serial.print(packet[3]);
    Serial.print(" z!");
    Serial.print(packet[4]);
    Serial.print(" c0_!");
    Serial.print(packet[5]);
    Serial.print(" c1_!");
    Serial.print(packet[6]);
    Serial.print(" c2_!");
    Serial.print(packet[7]);
    Serial.print(" c3_!");
    Serial.print(packet[8]);
    Serial.print("\n");
delay(100);
  
}
