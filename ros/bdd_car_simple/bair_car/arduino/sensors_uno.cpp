#include "constants.h"

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
//
void setup() {
    Serial.begin(115200);
    GPS_setup();
    gyro_setup();
    sonar_setup();
    //encoder_setup();
}

void loop() {
    GPS_loop();
    gyro_loop();
    sonar_loop();
    //encoder_loop();
}
//
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////





///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////// GPS /////////////////////////////////////////////////////////
//        ------> http://www.adafruit.com/products/746
//Adafruit ultimage GPS
//https://learn.adafruit.com/adafruit-ultimate-gps/arduino-wiring
//VIN to +5V
//GND to Ground
//RX to digital 2
//TX to digital 3
#include <Adafruit_GPS.h>
#include <SoftwareSerial.h>
SoftwareSerial mySerial(3, 2);
Adafruit_GPS GPS(&mySerial);
#define GPSECHO    false
boolean usingInterrupt = true; // Determine whether or not to use this
void useInterrupt(boolean); // Func prototype keeps Arduino 0023 happy
void GPS_setup()    
{
    //Serial.begin(115200);
    Serial.println("Adafruit GPS library basic test!");
    GPS.begin(9600);
    GPS.sendCommand(PMTK_SET_NMEA_OUTPUT_RMCGGA);
    GPS.sendCommand(PMTK_SET_NMEA_UPDATE_1HZ);     // 1 Hz update rate
    GPS.sendCommand(PGCMD_ANTENNA);
    useInterrupt(true);
    delay(1000);
    mySerial.println(PMTK_Q_RELEASE);
}
SIGNAL(TIMER0_COMPA_vect) {
    char c = GPS.read();
#ifdef UDR0
    if (GPSECHO)
        if (c) UDR0 = c;    
#endif
}
void useInterrupt(boolean v) {
    if (v) {
        OCR0A = 0xAF;
        TIMSK0 |= _BV(OCIE0A);
        usingInterrupt = true;
    } else {
        TIMSK0 &= ~_BV(OCIE0A);
        usingInterrupt = false;
    }
}
uint32_t timer = millis();
void GPS_loop()                                         // run over and over again
{
    if (! usingInterrupt) {
        char c = GPS.read();
        if (GPSECHO)
            if (c) Serial.print(c);
    }
    if (GPS.newNMEAreceived()) {
        if (!GPS.parse(GPS.lastNMEA()))     // this also sets the newNMEAreceived() flag to false
            return;    // we can fail to parse a sentence in which case we should just wait for another
    }
    if (timer > millis())    timer = millis();
    if (millis() - timer > 2000) { 
        timer = millis(); // reset the timer
        if (1){//(GPS.fix) {
            Serial.print("(");
            Serial.print(STATE_GPS);
            Serial.print(",");
            Serial.print(GPS.latitudeDegrees, 5);
            Serial.print(", "); 
            Serial.print(GPS.longitudeDegrees, 5);
            Serial.println(")"); 
        }
    }
}
//
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////







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
    //updateHeadings();
    Serial.print("(");
    Serial.print(STATE_GYRO);
    Serial.print(",");
    Serial.print(gyroDPS[0]);
    Serial.print(",");
    Serial.print(gyroDPS[1]);
    Serial.print(",");
    Serial.print(gyroDPS[2]);
    Serial.println(")");
    //printDPS();
    //Serial.print("     -->     ");
    //printHeadings();
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






///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////// max sonar /////////////////////////////////////////////////
// PINS: PW to pin 7
// Baud: ---
// http://playground.arduino.cc/Main/MaxSonar
//Author: Bruce Allen
//Digital pin 7 for reading in the pulse width from the MaxSonar device.
//This variable is a constant because the pin will not change throughout execution of this code.
const int pwPin = 7;
long pulse, inches, cm;
void sonar_setup()
{
    //This opens up a serial connection to shoot the results back to the PC console
    //Serial.begin(9600);
    ;
}
void sonar_loop()
{
    pinMode(pwPin, INPUT);
    pulse = pulseIn(pwPin, HIGH);
    inches = pulse / 147;
    cm = inches * 2.54;
    //Serial.print(inches);
    //Serial.print("in, ");
    //Serial.print(cm);
    Serial.print("(");
    Serial.print(STATE_SONAR);
    Serial.print(",");
    Serial.print(cm);
    Serial.println(")"); 
    //delay(500);
}
//////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////





///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
//////////////////// wheel encoder //////////////////////////////////////////////////
// PINS: 4,5 //2,3
// Baud: 9600
#define encoder0PinA    4 //2
#define encoder0PinB    5 //3
volatile int encoder0Pos = 0;
volatile boolean PastA = 0;
volatile boolean PastB = 0;
volatile unsigned long int a = 0;
volatile unsigned long int b = 0;
volatile unsigned long int t1 = millis();
volatile unsigned long int t2 = 0;
volatile unsigned long int dt = 0;
volatile float rate_1 = 0.0;
void encoder_setup() 
{
    //Serial.begin(9600);
    pinMode(encoder0PinA, INPUT);
    //turn on pullup resistor
    //digitalWrite(encoder0PinA, HIGH); //ONLY FOR SOME ENCODER(MAGNETIC)!!!! 
    pinMode(encoder0PinB, INPUT); 
    //turn on pullup resistor
    //digitalWrite(encoder0PinB, HIGH); //ONLY FOR SOME ENCODER(MAGNETIC)!!!! 
    PastA = (boolean)digitalRead(encoder0PinA); //initial value of channel A;
    PastB = (boolean)digitalRead(encoder0PinB); //and channel B
//To speed up even more, you may define manually the ISRs
// encoder A channel on interrupt 0 (arduino's pin 2)
    attachInterrupt(0, doEncoderA, CHANGE);
// encoder B channel pin on interrupt 1 (arduino's pin 3)
    attachInterrupt(1, doEncoderB, CHANGE); 
}
void encoder_loop()
{    
    dt = millis()-t1;
    if (dt > 100) {
        rate_1 = 1000.0/16.0 * a / dt;
        t1 = millis();
        a = 0;
    }
 //your staff....ENJOY! :D
    //Serial.print('(');
    //Serial.print(b);
    Serial.print("(");
    Serial.print(STATE_ENCODER);
    Serial.print(",");
    Serial.println(rate_1);
    Serial.println(")");
    //delay(100);
}
//you may easily modify the code    get quadrature..
//..but be sure this whouldn't let Arduino back! 
void doEncoderA()
{
         t2 = micros();
         a = a + 1;
}
void doEncoderB()
{
         b += 1;
}
//////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////



