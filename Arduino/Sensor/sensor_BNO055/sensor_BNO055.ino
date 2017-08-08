#include <Adafruit_BNO055.h>
#include <SoftwareSerial.h>
#include <EEPROM.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include "constants.h"

/* Set the delay between fresh samples */
#define BNO055_SAMPLERATE_DELAY_MS (10)

Adafruit_BNO055 bno = Adafruit_BNO055(20);

/**************************************************************************/
/*
    Displays some basic information on this sensor from the unified
    sensor API sensor_t type (see Adafruit_Sensor for more information)
    */
/**************************************************************************/
void displaySensorDetails(void)
{
    sensor_t sensor;
    bno.getSensor(&sensor);
    Serial.println("------------------------------------");
    Serial.print("Sensor:       "); Serial.println(sensor.name);
    Serial.print("Driver Ver:   "); Serial.println(sensor.version);
    Serial.print("Unique ID:    "); Serial.println(sensor.sensor_id);
    Serial.print("Max Value:    "); Serial.print(sensor.max_value); Serial.println(" xxx");
    Serial.print("Min Value:    "); Serial.print(sensor.min_value); Serial.println(" xxx");
    Serial.print("Resolution:   "); Serial.print(sensor.resolution); Serial.println(" xxx");
    Serial.println("------------------------------------");
    Serial.println("");
    delay(500);
}

/**************************************************************************/
/*
    Display some basic info about the sensor status
    */
/**************************************************************************/
void displaySensorStatus(void)
{
    /* Get the system status values (mostly for debugging purposes) */
    uint8_t system_status, self_test_results, system_error;
    system_status = self_test_results = system_error = 0;
    bno.getSystemStatus(&system_status, &self_test_results, &system_error);

    /* Display the results in the Serial Monitor */
    Serial.println("");
    Serial.print("System Status: 0x");
    Serial.println(system_status, HEX);
    Serial.print("Self Test:     0x");
    Serial.println(self_test_results, HEX);
    Serial.print("System Error:  0x");
    Serial.println(system_error, HEX);
    Serial.println("");
    delay(500);
}

/**************************************************************************/
/*
    Display sensor calibration status
    */
/**************************************************************************/
void displayCalStatus(void)
{
    /* Get the four calibration values (0..3) */
    /* Any sensor data reporting 0 should be ignored, */
    /* 3 means 'fully calibrated" */
    uint8_t system, gyro, accel, mag;
    system = gyro = accel = mag = 0;
    bno.getCalibration(&system, &gyro, &accel, &mag);

    /* The data should be ignored until the system calibration is > 0 */
    Serial.print("\t");
    if (!system)
    {
        Serial.print("! ");
    }

    /* Display the individual values */
    Serial.print("Sys:");
    Serial.print(system, DEC);
    Serial.print(" G:");
    Serial.print(gyro, DEC);
    Serial.print(" A:");
    Serial.print(accel, DEC);
    Serial.print(" M:");
    Serial.print(mag, DEC);
}

/**************************************************************************/
/*
    Display the raw calibration offset and radius data
    */
/**************************************************************************/
void displaySensorOffsets(const adafruit_bno055_offsets_t &calibData)
{
    Serial.print("Accelerometer: ");
    Serial.print(calibData.accel_offset_x); Serial.print(" ");
    Serial.print(calibData.accel_offset_y); Serial.print(" ");
    Serial.print(calibData.accel_offset_z); Serial.print(" ");

    Serial.print("\nGyro: ");
    Serial.print(calibData.gyro_offset_x); Serial.print(" ");
    Serial.print(calibData.gyro_offset_y); Serial.print(" ");
    Serial.print(calibData.gyro_offset_z); Serial.print(" ");

    Serial.print("\nMag: ");
    Serial.print(calibData.mag_offset_x); Serial.print(" ");
    Serial.print(calibData.mag_offset_y); Serial.print(" ");
    Serial.print(calibData.mag_offset_z); Serial.print(" ");

    Serial.print("\nAccel Radius: ");
    Serial.print(calibData.accel_radius);

    Serial.print("\nMag Radius: ");
    Serial.print(calibData.mag_radius);
}


/**************************************************************************/
/*
    Arduino setup function (automatically called at startup)
    */
/**************************************************************************/
void setup(void)
{
    Serial.begin(115200);
    delay(1000);
    //Serial.println("Orientation Sensor Test"); Serial.println("");

    /* Initialise the sensor */
    if (!bno.begin())
    {
        /* There was a problem detecting the BNO055 ... check your connections */
        Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
        while (1);
    }

    int eeAddress = 0;
    long bnoID;
    bool foundCalib = false;

    EEPROM.get(eeAddress, bnoID);

    adafruit_bno055_offsets_t calibrationData;
    sensor_t sensor;

    /*
    *  Look for the sensor's unique ID at the beginning oF EEPROM.
    *  This isn't foolproof, but it's better than nothing.
    */
    bno.getSensor(&sensor);
    if (bnoID != sensor.sensor_id)
    {
        Serial.println("\nNo Calibration Data for this sensor exists in EEPROM");
        delay(500);
    }
    else
    {
        //Serial.println("\nFound Calibration for this sensor in EEPROM.");
        eeAddress += sizeof(long);
        EEPROM.get(eeAddress, calibrationData);

        //displaySensorOffsets(calibrationData);

        //Serial.println("\n\nRestoring Calibration data to the BNO055...");
        bno.setSensorOffsets(calibrationData);

        //Serial.println("\n\nCalibration data loaded into BNO055");
        foundCalib = true;
    }

    delay(1000);

    /* Display some basic information on this sensor */
    //displaySensorDetails();

    /* Optional: Display current status */
    //displaySensorStatus();

    bno.setExtCrystalUse(true);

    sensors_event_t event;
    bno.getEvent(&event);
    if (foundCalib){
        /*
        Serial.println("Move sensor slightly to calibrate magnetometers");
        while (!bno.isFullyCalibrated())
        {
            bno.getEvent(&event);
            delay(BNO055_SAMPLERATE_DELAY_MS);
        }
        */
    }
    else
    {
        Serial.println("Please Calibrate Sensor: ");
        while (!bno.isFullyCalibrated())
        {
            bno.getEvent(&event);

            Serial.print("X: ");
            Serial.print(event.orientation.x, 4);
            Serial.print("\tY: ");
            Serial.print(event.orientation.y, 4);
            Serial.print("\tZ: ");
            Serial.print(event.orientation.z, 4);

            /* Optional: Display calibration status */
            displayCalStatus();

            /* New line for the next sample */
            Serial.println("");

            /* Wait the specified delay before requesting new data */
            delay(BNO055_SAMPLERATE_DELAY_MS);
        }
    }
    /*
    Serial.println("\nFully calibrated!");
    Serial.println("--------------------------------");
    Serial.println("Calibration Results: ");
    */
    adafruit_bno055_offsets_t newCalib;
    bno.getSensorOffsets(newCalib);
    /*
    displaySensorOffsets(newCalib);

    Serial.println("\n\nStoring calibration data to EEPROM...");
    */
    eeAddress = 0;
    bno.getSensor(&sensor);
    bnoID = sensor.sensor_id;

    EEPROM.put(eeAddress, bnoID);

    eeAddress += sizeof(long);
    EEPROM.put(eeAddress, newCalib);
    /*
    Serial.println("Data stored to EEPROM.");

    Serial.println("\n--------------------------------\n");
    */
    delay(500);
}

// Possible vector values can be:
// - VECTOR_ACCELEROMETER - m/s^2
// - VECTOR_MAGNETOMETER  - uT
// - VECTOR_GYROSCOPE     - rad/s
// - VECTOR_EULER         - degrees
// - VECTOR_LINEARACCEL   - m/s^2
// - VECTOR_GRAVITY       - m/s^2
// Since we use the 9-DOF fusion mode, the sensitivity is automatic controlled by the mcu.
const float pi = 3.14159;
double accValues[3] = {0.0f};  //accValues[x], accValues[y], accValues[z]
double gyroValues[3] = {0.0f};
double orientation[3] = {0.0f};
double head[3] = {0.0f};  

void updateAccValues() {
    imu::Vector<3> acc = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
    accValues[0] = acc.x();
    accValues[1] = acc.y();
    accValues[2] = acc.z();
}

void updateGyroValues() {
    imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    gyroValues[0] = gyro.x();
    gyroValues[1] = gyro.y();
    gyroValues[2] = gyro.z();
}

void updateOrientation() {
    imu::Quaternion quat = bno.getQuat();
    imu::Vector<3> oriet = quat.toEuler();
    orientation[0] = oriet.x();
    orientation[1] = oriet.y();
    orientation[2] = oriet.z();
}

void updateHeadings() {  
    float deltaT=getDeltaTMicros();
    for (int j=0;j<3;j++) {
        head[j] += (gyroValues[j]*deltaT)/1000000.0f;
    }
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

void printAccValues() {
    Serial.print("('acc',");
    Serial.print(accValues[0]); Serial.print(",");
    Serial.print(accValues[1]); Serial.print(",");
    Serial.print(accValues[2]); Serial.println(")");
}

void printHeadings() {
    Serial.print("('head',");
    Serial.print(head[0]);
    Serial.print(',');

    Serial.print(head[1]);
    Serial.print(',');

    Serial.print(head[2]);
    Serial.println(')');
}

void printGyroValues() {
    Serial.print("(");
    Serial.print(STATE_GYRO);
    Serial.print(",");
    Serial.print(gyroValues[0]);
    Serial.print(",");
    Serial.print(gyroValues[1]);
    Serial.print(",");
    Serial.print(gyroValues[2]);
    Serial.println(")");
}

void printOrientation() {
    Serial.print("(");
    Serial.print("'head'");
    Serial.print(",");
    Serial.print(orientation[0]);
    Serial.print(",");
    Serial.print(orientation[1]);
    Serial.print(",");
    Serial.print(orientation[2]);
    Serial.println(")");
}

void loop() {
    updateAccValues();
    updateGyroValues();
    updateOrientation();
    ////updateHeadings();
    printAccValues();
    printGyroValues();
    ////printHeadings();
    printOrientation();
    /*
    sensors_event_t event;
    bno.getEvent(&event);
    Serial.print("X: ");
    Serial.print(event.orientation.x, 4);
    Serial.print("\tY: ");    
    Serial.print(event.orientation.y, 4);
    Serial.print("\tZ: ");
    Serial.print(event.orientation.z, 4);
    */
    
    /* Optional: Display calibration status */
    //displayCalStatus();

    /* Optional: Display sensor status (debug only) */
    //displaySensorStatus();

    /* New line for the next sample */
    Serial.println("");

    /* Wait the specified delay before requesting new data */
    delay(BNO055_SAMPLERATE_DELAY_MS);
}
