#include <SoftwareSerial.h>
#include <Wire.h>
#include <Adafruit_MMA8451.h>
#include <Adafruit_Sensor.h>
#include "constants.h"

Adafruit_MMA8451 mma = Adafruit_MMA8451();


void setup()    
{
    
    Serial.begin(115200);
    gyro_setup();
    //Serial.println("Adafruit MMA8451 test!");
    if (! mma.begin()) {
        //Serial.println("Couldnt start");
        while (1);
    }
    //Serial.println("MMA8451 found!");
    mma.setRange(MMA8451_RANGE_2_G);
    //Serial.print("Range = "); Serial.print(2 << mma.getRange());    
    //Serial.println("G");
    
}



//////////////////////////////////////////////////////

float ax = 0;
float ay = 0;
float az = 0;
float ctr = 0;

uint32_t timer = millis();
void loop() {
    
    //gyro_loop();
/*        
    if (timer > millis())    timer = millis();

    if (millis() - timer > 1000) {
         timer = millis();
    }
    
    sensors_event_t event; 
    mma.getEvent(&event);
    Serial.print("(acc,");
    Serial.print(event.acceleration.x); Serial.print(",");
    Serial.print(event.acceleration.y); Serial.print(",");
    Serial.print(event.acceleration.z); Serial.print(")");
    Serial.println();
    delay(1000/100);
*/
    sensors_event_t event; 
    mma.getEvent(&event);
    ax += event.acceleration.x;
    ay += event.acceleration.y;
    az += event.acceleration.z;
    ctr += 1;


    if (timer > millis())    timer = millis();

    if (millis() - timer > 10) {
        timer = millis();
        /*
        Serial.print("('acc',");
        Serial.print(ax/ctr); Serial.print(",");
        Serial.print(ay/ctr); Serial.print(",");
        Serial.print(az/ctr); Serial.print(")");
        //Serial.print(ctr); Serial.print(")");
        Serial.println();
        */
        ax = 0;
        ay = 0;
        az = 0;
        ctr = 0;
        gyro_loop();
    }


}







///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
//////////// gyroscope //////////////////////////////////////////////////////////
// PINS: A4,A5
// Baud: 115200
// http://forum.arduino.cc/index.php?topic=147351.0
// http://www.livescience.com/40103-accelerometer-vs-gyroscope.html
//Parallax Gyroscope Module 3-Axis L3G4200D
//https://www.parallax.com/product/27911
//http://forum.arduino.cc/index.php?topic=147351.msg1106879#msg1106879
//VIN to +5V
//GND to Ground
//SCL line to pin A5
//SDA line to pin A4
#include <Wire.h>
#define    CTRL_REG1    0x20
#define    CTRL_REG2    0x21
#define    CTRL_REG3    0x22
#define    CTRL_REG4    0x23
#define    CTRL_REG5    0x24
#define    CTRL_REG6    0x25
int gyroI2CAddr=105;
int gyroRaw[3];                                                 // raw sensor data, each axis, pretty useless really but here it is.
double gyroDPS[3];                                            // gyro degrees per second, each axis
float heading[3]={0.0f};                                // heading[x], heading[y], heading [z]
int gyroZeroRate[3];                                        // Calibration data.    Needed because the sensor does center at zero, but rather always reports a small amount of rotation on each axis.
int gyroThreshold[3];                                     // Raw rate change data less than the statistically derived threshold is discarded.
#define    NUM_GYRO_SAMPLES    50                     // As recommended in STMicro doc
#define    GYRO_SIGMA_MULTIPLE    3                 // As recommended 
float dpsPerDigit=.00875f;                            // for conversion to degrees per second
void gyro_setup() {
    //Serial.begin(115200);
    Wire.begin();
    setupGyro();
    calibrateGyro();
}
void gyro_loop() {
    updateGyroValues();
    updateHeadings();
    /*
    Serial.print("(");
    Serial.print(STATE_GYRO);
    Serial.print(",");
    Serial.print(gyroDPS[0]);
    Serial.print(",");
    Serial.print(gyroDPS[1]);
    Serial.print(",");
    Serial.print(gyroDPS[2]);
    Serial.println(")");
    */
    //printDPS();
    //Serial.print("     -->     ");
    printHeadings2();
    //Serial.println();
}
void printDPS()
{
    Serial.print("DPS X: ");
    Serial.print(gyroDPS[0]);
    Serial.print("    Y: ");
    Serial.print(gyroDPS[1]);
    Serial.print("    Z: ");
    Serial.print(gyroDPS[2]);
}
void printHeadings()
{
    Serial.print("Heading X: ");
    Serial.print(heading[0]);
    Serial.print("    Y: ");
    Serial.print(heading[1]);
    Serial.print("    Z: ");
    Serial.print(heading[2]);
}
void printHeadings2()
{
    Serial.print("('head',");
    Serial.print(heading[0]);
    Serial.print(',');

    Serial.print(heading[1]);
    Serial.print(',');

    Serial.print(heading[2]);
    Serial.println(')');
}

void updateHeadings()
{
    float deltaT=getDeltaTMicros();
    for (int j=0;j<3;j++)
        heading[j] -= (gyroDPS[j]*deltaT)/1000000.0f;
}
unsigned long getDeltaTMicros()
{
    static unsigned long lastTime=0;
    unsigned long currentTime=micros();
    unsigned long deltaT=currentTime-lastTime;
    if (deltaT < 0.0)
         deltaT=currentTime+(4294967295-lastTime);
    lastTime=currentTime;
    return deltaT;
}
void testCalibration()
{
    calibrateGyro();
    for (int j=0;j<3;j++)
    {
        Serial.print(gyroZeroRate[j]);
        Serial.print("    ");
        Serial.print(gyroThreshold[j]);
        Serial.print("    ");    
    }
    Serial.println();
    return; 
}
void setupGyro()
{
    gyroWriteI2C(CTRL_REG1, 0x1F);                // Turn on all axes, disable power down
    gyroWriteI2C(CTRL_REG3, 0x08);                // Enable control ready signal
    setGyroSensitivity500();
    delay(100);
}
void calibrateGyro()
{
    long int gyroSums[3]={0};
    long int gyroSigma[3]={0};
    for (int i=0;i<NUM_GYRO_SAMPLES;i++)
    {
        updateGyroValues();
        for (int j=0;j<3;j++)
        {
            gyroSums[j]+=gyroRaw[j];
            gyroSigma[j]+=gyroRaw[j]*gyroRaw[j];
        }
    }
    for (int j=0;j<3;j++)
    {
        int averageRate=gyroSums[j]/NUM_GYRO_SAMPLES;
        gyroZeroRate[j]=averageRate;
        gyroThreshold[j]=sqrt((double(gyroSigma[j]) / NUM_GYRO_SAMPLES) - (averageRate * averageRate)) * GYRO_SIGMA_MULTIPLE;        
    }
}
void updateGyroValues() {
    while (!(gyroReadI2C(0x27) & B00001000)){}            // Without this line you will get bad data occasionally
    int reg=0x28;
    for (int j=0;j<3;j++)
    {
        gyroRaw[j]=(gyroReadI2C(reg) | (gyroReadI2C(reg+1)<<8));
        reg+=2;
    }
    int deltaGyro[3];
    for (int j=0;j<3;j++)
    {
        deltaGyro[j]=gyroRaw[j]-gyroZeroRate[j];            // Use the calibration data to modify the sensor value.
        if (abs(deltaGyro[j]) < gyroThreshold[j])
            deltaGyro[j]=0;
        gyroDPS[j]= dpsPerDigit * deltaGyro[j];            // Multiply the sensor value by the sensitivity factor to get degrees per second.
    }
}
void setGyroSensitivity250(void)
{
    dpsPerDigit=.00875f;
    gyroWriteI2C(CTRL_REG4, 0x80);                // Set scale (250 deg/sec)
}
void setGyroSensitivity500(void)
{
    dpsPerDigit=.0175f;
    gyroWriteI2C(CTRL_REG4, 0x90);                // Set scale (500 deg/sec)
}
void setGyroSensitivity2000(void)
{
    dpsPerDigit=.07f;
    gyroWriteI2C(CTRL_REG4,0xA0); 
}
int gyroReadI2C (byte regAddr) {
    Wire.beginTransmission(gyroI2CAddr);
    Wire.write(regAddr);
    Wire.endTransmission();
    Wire.requestFrom(gyroI2CAddr, 1);
    while(!Wire.available()) {};
    return (Wire.read());
}
int gyroWriteI2C( byte regAddr, byte val){
    Wire.beginTransmission(gyroI2CAddr);
    Wire.write(regAddr);
    Wire.write(val);
    Wire.endTransmission();
}
//
//////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


