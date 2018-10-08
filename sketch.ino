// 1) Install https://github.com/hanyazou/BMI160-Arduino as ZIP library
// 2) Connect in I2C mode

#include <BMI160Gen.h>
 
float pitch = 0.0;
float roll = 0.0;
int printCounter = 0;

#define GYROSCOPE_SENSITIVITY 65.536
#define DT 0.01

typedef struct {
  int x;
  int y;
  int z;
} vec3_t;

vec3_t toVec3(int x, int y, int z) {
  return (vec3_t){.x = x, .y = y, .z = z};
}

void clamp(float &n, float lo, float hi) {
  if(n < lo) { n = lo; return; }
  if(n > hi) { n = hi; return; }
}

// idea from http://www.pieter-jan.com/node/11
void compFilter(vec3_t accData, vec3_t gyrData, float &pitch, float &roll)
{
  float pitchAcc, rollAcc;               

  // Dead reckoning for fast changes
  roll += ((float)gyrData.x / GYROSCOPE_SENSITIVITY) * DT;
  pitch -= ((float)gyrData.y / GYROSCOPE_SENSITIVITY) * DT;

  // Accelerometer at steady state
  rollAcc = atan2f((float)accData.y, (float)accData.z) * 180 / M_PI;
  roll = roll * 0.98 + rollAcc * 0.02;

  pitchAcc = atan2f((float)accData.x, (float)accData.z) * 180 / M_PI;
  pitch = pitch * 0.98 + pitchAcc * 0.02;
} 
 
void setup() {
  Serial.begin(9600);
  while (!Serial);

  BMI160.begin(BMI160GenClass::I2C_MODE, 0x69);
}
 
void loop() {
  int ax, ay, az;
  int gx, gy, gz;
  vec3_t aVec, gVec;

  BMI160.readMotionSensor(ax, ay, az, gx, gy, gz);
  aVec = toVec3(ax, ay, az);
  gVec = toVec3(gx, gy, gz);
  compFilter(aVec, gVec, pitch, roll);
  clamp(pitch, -90.0, 90.0);
  clamp(roll, -180.0, 180.0);

  printCounter++;
  printCounter %= (int)(1/(4*DT));
  if(printCounter == 0) {
    Serial.print("pr: ");
    Serial.print(pitch);
    Serial.print("\t");
    Serial.print(roll);
    Serial.print("\n");
  }
 
  delay((int)(DT * 1000));
}
