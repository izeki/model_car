///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
//
void setup() {
    Serial.begin(115200);
    GPS_setup();
    gyro_setup();
    //motor_servo_setup();
    sonar_setup();
    //encoder_setup();
}

void loop() {
    GPS_loop();
    gyro_loop();
    //motor_servo_loop();
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
            Serial.print("(-1,");
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
///////////// motor_servo /////////////////////////////////////////////////
// PINS: 8,9,10,11,12,13
// Baud: 115200
//
//PIN_SERVO_IN 11
//PIN_MOTOR_IN 10
//PIN_BUTTON_IN 12 (or 8)
//PIN_SERVO_OUT 9
//PIN_MOTOR_OUT 8 (or 12)
//Code written for Arduino Uno. This code conflicts with the Adafruit GPS code.
//The purpose is to read PWM signals coming out of a radio receiver
//and either relay them unchanged to ESC/servo or substitute signals from a host system.
//The steering servo and ESC(motor) are controlled with PWM signals. These meaning of these signals
//may vary with time, and across devices. To get uniform behavior, the user calibrates with each session.
//The host system deals only in percent control signals that are assumed to be based on calibrated PWM
//signals. Thus, 0 should always mean 'extreme left', 49 should mean 'straight ahead', and 99 'extreme right'
//in the percent signals, whereas absolute values of the PWM can vary for various reasons.
//24 April 2016
#include "PinChangeInterrupt.h" // Adafruit library
#include <Servo.h> // Arduino library
// These come from the radio receiver via three black-red-white ribbons.
#define PIN_SERVO_IN 11
#define PIN_MOTOR_IN 10
#define PIN_BUTTON_IN 12
// These go out to ESC (motor controller) and steer servo via black-red-white ribbons.
#define PIN_SERVO_OUT 9
#define PIN_MOTOR_OUT 8
// On-board LED, used to signal error state
#define PIN_LED_OUT 13
// These define extreme min an max values that should never be broken.
#define SERVO_MAX     2000
#define MOTOR_MAX     SERVO_MAX
#define BUTTON_MAX    SERVO_MAX
#define SERVO_MIN     500
#define MOTOR_MIN     SERVO_MIN
#define BUTTON_MIN    SERVO_MIN
// These are the possible states of the control system.
// States are reached by button presses or drive commands, except for error state.
#define STATE_HUMAN_FULL_CONTROL                        1
#define STATE_LOCK                                                    2
#define STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR 3
#define STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR 5
#define STATE_LOCK_CALIBRATE                                4
#define STATE_ERROR                                                 -1
//////////////
// Below are variables that hold ongoing signal data. I try to initalize them to
// to sensible values, but they will immediately be reset in the running program.
//
// These volatile variables are set by interrupt service routines
// tied to the servo and motor input pins. These values will be reset manually in the
// STATE_LOCK_CALIBRATE state, so conservative values are given here.
volatile int servo_null_pwm_value = 1500;
volatile int servo_max_pwm_value    = 1600;
volatile int servo_min_pwm_value    = 1400;
volatile int motor_null_pwm_value = 1528;
volatile int motor_max_pwm_value    = 1600;
volatile int motor_min_pwm_value    = 1400;
// These are three key values indicating current incoming signals.
// These are set in interrupt service routines.
volatile int button_pwm_value = 1210;
volatile int servo_pwm_value = servo_null_pwm_value;
volatile int motor_pwm_value = motor_null_pwm_value;
// These are used to interpret interrupt signals.
volatile long int button_prev_interrupt_time = 0;
volatile long int servo_prev_interrupt_time    = 0;
volatile long int motor_prev_interrupt_time    = 0;
volatile long int state_transition_time_ms = 0;
// Some intial conditions, putting the system in lock state.
volatile int state = STATE_LOCK;
volatile int previous_state = 0;
// Variable to receive caffe data and format it for output.
long int caffe_last_int_read_time;
int caffe_mode = -3;
int caffe_servo_percent = 49;
int caffe_motor_percent = 49;
int caffe_servo_pwm_value = servo_null_pwm_value;
int caffe_motor_pwm_value = motor_null_pwm_value;
// The host computer is not to worry about PWM values. These variables hold percent values
// that are passed up to host.
int servo_percent = 49;
int motor_percent = 49;
//
/////////////////////
// Servo classes. ESC (motor) is treated as a servo for signaling purposes.
Servo servo;
Servo motor; 
////////////////////////////////////////
//
void motor_servo_setup()
{
    // Establishing serial communication with host system. The best values for these parameters
    // is an open question. At 9600 baud rate, data can be missed.
    //Serial.begin(115200);
    Serial.setTimeout(5);
    // Setting up three input pins
    pinMode(PIN_BUTTON_IN, INPUT_PULLUP);
    pinMode(PIN_SERVO_IN, INPUT_PULLUP);
    pinMode(PIN_MOTOR_IN, INPUT_PULLUP);
    // LED out
    pinMode(PIN_LED_OUT, OUTPUT);
    // Attach interrupt service routines to pins. A change in signal triggers interrupts.
    attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(PIN_BUTTON_IN),
        button_interrupt_service_routine, CHANGE);
    attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(PIN_SERVO_IN),
        servo_interrupt_service_routine, CHANGE);
    attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(PIN_MOTOR_IN),
        motor_interrupt_service_routine, CHANGE);
    // Attach output pins to ESC (motor) and steering servo.
    servo.attach(PIN_SERVO_OUT); 
    motor.attach(PIN_MOTOR_OUT); 
}
////////////////////////////////////////
//
void motor_servo_loop() {
    check_for_error_conditions();
    // Try to read the "caffe_int" sent by the host system (there is a timeout on serial reads, so the Arduino
    // doesn't wait long to get one -- in which case the caffe_int is set to zero.)
    int caffe_int = Serial.parseInt();
    // If it is received, decode it to yield three control values (i.e., caffe_mode, caffe_servo_percent, and caffe_motor_percent)
    if (caffe_int > 0) {
            caffe_last_int_read_time = micros()/1000;
            caffe_mode = caffe_int/10000;
            caffe_servo_percent = (caffe_int-caffe_mode*10000)/100;
            caffe_motor_percent = (caffe_int-caffe_servo_percent*100-caffe_mode*10000);
        } else if (caffe_int < 0) {
            caffe_mode = caffe_int/10000;
    }
    // Turn the caffe_servo_percent and caffe_motor_percent values from 0 to 99 values into PWM values that can be sent 
    // to ESC (motor) and servo.
    if (caffe_mode > 0) {
        if (caffe_servo_percent >= 49) {
            caffe_servo_pwm_value = (caffe_servo_percent-49)/50.0 * (servo_max_pwm_value - servo_null_pwm_value) + servo_null_pwm_value;
        }
        else {
            caffe_servo_pwm_value = (caffe_servo_percent - 50)/50.0 * (servo_null_pwm_value - servo_min_pwm_value) + servo_null_pwm_value;
        }
        if (caffe_motor_percent >= 49) {
            caffe_motor_pwm_value = (caffe_motor_percent-49)/50.0 * (motor_max_pwm_value - motor_null_pwm_value) + motor_null_pwm_value;
        }
        else {
            caffe_motor_pwm_value = (caffe_motor_percent - 50)/50.0 * (motor_null_pwm_value - motor_min_pwm_value) + motor_null_pwm_value;
        }
    }
    else {
        caffe_servo_pwm_value = servo_null_pwm_value;
    }
    // Compute command signal percents from signals from the handheld radio controller
    // to be sent to host computer, which doesn't bother with PWM values
    if (servo_pwm_value >= servo_null_pwm_value) {
        servo_percent = 49+50.0*(servo_pwm_value-servo_null_pwm_value)/(servo_max_pwm_value-servo_null_pwm_value);
    }
    else {
        servo_percent = 49 - 49.0*(servo_null_pwm_value-servo_pwm_value)/(servo_null_pwm_value-servo_min_pwm_value);
    }
    if (motor_pwm_value >= motor_null_pwm_value) {
        motor_percent = 49+50.0*(motor_pwm_value-motor_null_pwm_value)/(motor_max_pwm_value-motor_null_pwm_value);
    }
    else {
        motor_percent = 49 - 49.0*(motor_null_pwm_value-motor_pwm_value)/(motor_null_pwm_value-motor_min_pwm_value);
    }
    int debug = false;
    if (debug) {
        Serial.print("(");
        Serial.print(state);
        Serial.print(",[");
        Serial.print(button_pwm_value);
        Serial.print(",");
        Serial.print(servo_pwm_value);        
        Serial.print(",");
        Serial.print(motor_pwm_value);
        Serial.print("],[");
        Serial.print(servo_min_pwm_value);
        Serial.print(",");
        Serial.print(servo_null_pwm_value);
        Serial.print(",");
        Serial.print(servo_max_pwm_value);    
        Serial.print("],[");
        Serial.print(motor_min_pwm_value);
        Serial.print(",");
        Serial.print(motor_null_pwm_value);
        Serial.print(",");
        Serial.print(motor_max_pwm_value);
        Serial.print("],");
        Serial.print(caffe_mode);
        Serial.print(",");
        Serial.print(caffe_servo_percent);
        Serial.print(",");
        Serial.print(caffe_motor_percent);
        Serial.print(",");
        Serial.print(caffe_servo_pwm_value);
        Serial.print(",");
        Serial.print(caffe_motor_pwm_value);
        Serial.print(",");
        Serial.print(servo_percent);
        Serial.print(",");
        Serial.print(motor_percent);
        Serial.print(",");
        Serial.print(millis() - state_transition_time_ms);
        Serial.println(")");
    }
    else {
        // Send data string which looks like a python tuple.
        Serial.print("(-2,");
        Serial.print(state);
        Serial.print(",");
        Serial.print(servo_percent);
        Serial.print(",");
        Serial.print(motor_percent);
        Serial.print(",");
        Serial.print((millis() - state_transition_time_ms)/1000); //one second resolution
        Serial.println(")");
    }
    delay(10); // How long should this be? Note, this is in ms, whereas most other times are in micro seconds.
    // Blink LED if in error state.
    if (state == STATE_ERROR) {
        digitalWrite(PIN_LED_OUT, HIGH);
        delay(100);
        digitalWrite(PIN_LED_OUT, LOW);
        delay(100);
    }
}
//
////////////////////////////////////////
////////////////////////////////////////
//
// The hand-held radio controller has two buttons. Pressing the upper or lower
// allows for reaching separate PWM levels: ~ 1710, 1200, 1000, and 888
// These are used for different control states.
#define BUTTON_A 1710 // Human in full control of driving
#define BUTTON_B 1200 // Lock state
#define BUTTON_C 964    // Caffe steering, human on accelerator
#define BUTTON_D 850    // Calibration of steering and motor control ranges
#define BUTTON_DELTA 50 // range around button value that is considered in that value
void button_interrupt_service_routine(void) {
    volatile long int m = micros();
    volatile long int dt = m - button_prev_interrupt_time;
    button_prev_interrupt_time = m;
    // Human in full control of driving
    if (dt>BUTTON_MIN && dt<BUTTON_MAX) {
        button_pwm_value = dt;
        if (abs(button_pwm_value-BUTTON_A)<BUTTON_DELTA) {
            if (state == STATE_ERROR) return;
            if (state != STATE_HUMAN_FULL_CONTROL) {
                previous_state = state;
                state = STATE_HUMAN_FULL_CONTROL;
                state_transition_time_ms = m/1000.0;
            }
        }
        // Lock state
        else if (abs(button_pwm_value-BUTTON_B)<BUTTON_DELTA) {
            if (state == STATE_ERROR) return;
            if (state != STATE_LOCK) {
                previous_state = state;
                state = STATE_LOCK;
                state_transition_time_ms = m/1000.0;
            }
        }
        // Caffe steering, human on accelerator
        else if (abs(button_pwm_value-BUTTON_C)<BUTTON_DELTA) {
            if (state == STATE_ERROR) return;
            if (state != STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR && state != STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR) {
                previous_state = state;
                state = STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR;
                state_transition_time_ms = m/1000.0;
            }
        }
        // Calibration of steering and motor control ranges
        else if (abs(button_pwm_value-BUTTON_D)<BUTTON_DELTA) {
            if (state != STATE_LOCK_CALIBRATE) {
                previous_state = state;
                state = STATE_LOCK_CALIBRATE;
                state_transition_time_ms = m/1000.0;
                servo_null_pwm_value = servo_pwm_value;
                servo_max_pwm_value = servo_null_pwm_value;
                servo_min_pwm_value = servo_null_pwm_value;
                motor_null_pwm_value = motor_pwm_value;
                motor_max_pwm_value = motor_null_pwm_value;
                motor_min_pwm_value = motor_null_pwm_value;
                caffe_servo_pwm_value = servo_null_pwm_value;
                caffe_motor_pwm_value = motor_null_pwm_value;
            }
            if (servo_pwm_value > servo_max_pwm_value) {
                servo_max_pwm_value = servo_pwm_value;
            }
            if (servo_pwm_value < servo_min_pwm_value) {
                servo_min_pwm_value = servo_pwm_value;
            }
            if (motor_pwm_value > motor_max_pwm_value) {
                motor_max_pwm_value = motor_pwm_value;
            }
            if (motor_pwm_value < motor_min_pwm_value) {
                motor_min_pwm_value = motor_pwm_value;
            }
        }
    }
}
//
////////////////////////////////////////
////////////////////////////////////////
//
// Servo interrupt service routine. This would be very short except that the human can take
// control from Caffe, and Caffe can take back control if steering left in neutral position.
void servo_interrupt_service_routine(void) {
    volatile long int m = micros();
    volatile long int dt = m - servo_prev_interrupt_time;
    servo_prev_interrupt_time = m;
    if (state == STATE_ERROR) return; // no action if in error state
    if (dt>SERVO_MIN && dt<SERVO_MAX) {
        servo_pwm_value = dt;
        if (state == STATE_HUMAN_FULL_CONTROL) {
            servo.writeMicroseconds(servo_pwm_value);
        }
        else if (state == STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR) {
            // If steer is close to neutral, let Caffe take over.
            if (abs(servo_pwm_value-servo_null_pwm_value)<=30 ){
                previous_state = state;
                state = STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR;
                state_transition_time_ms = m/1000.0;
                //servo.writeMicroseconds((caffe_servo_pwm_value+servo_pwm_value)/2);
            }
            else {
                servo.writeMicroseconds(servo_pwm_value);
            }
        }
        // If human makes strong steer command, let human take over.
        else if (state == STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR) {
            if (abs(servo_pwm_value-servo_null_pwm_value)>70) {
                previous_state = state;
                state = STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR;
                state_transition_time_ms = m/1000.0;
                //servo.writeMicroseconds(servo_pwm_value);     
            }
            else {
                servo.writeMicroseconds(caffe_servo_pwm_value);
            }
        }
        else {
            ;//servo.writeMicroseconds(servo_null_pwm_value);
        }
    } 
}
//
////////////////////////////////////////
////////////////////////////////////////
//
// Motor interrupt service routine. This is simple because right now only human controls motor.
void motor_interrupt_service_routine(void) {
    volatile long int m = micros();
    volatile long int dt = m - motor_prev_interrupt_time;
    motor_prev_interrupt_time = m;
    // Locking out in error state has bad results -- cannot switch off motor manually if it is on.
    // if (state == STATE_ERROR) return;
    if (dt>MOTOR_MIN && dt<MOTOR_MAX) {
        motor_pwm_value = dt;
        if (state == STATE_HUMAN_FULL_CONTROL) {
            motor.writeMicroseconds(motor_pwm_value);
        }
        else if (state == STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR) {
            motor.writeMicroseconds(motor_pwm_value);
        }
        else if (state == STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR) {
            motor.writeMicroseconds(motor_pwm_value);
        }
        else {
            ;//motor.writeMicroseconds(motor_null_pwm_value);
        }
    } 
}
//
////////////////////////////////////////
////////////////////////////////////////
//
int check_for_error_conditions(void) {
// Check state of all of these variables for out-of-bound conditions
    // If in calibration state, ignore potential errors in order to attempt to correct.
    if (state == STATE_LOCK_CALIBRATE) return(1);
    if (
        safe_pwm_range(servo_null_pwm_value) &&
        safe_pwm_range(servo_max_pwm_value) &&
        safe_pwm_range(servo_min_pwm_value) &&
        safe_pwm_range(motor_null_pwm_value) &&
        safe_pwm_range(motor_max_pwm_value) &&
        safe_pwm_range(motor_min_pwm_value) &&
        safe_pwm_range(servo_pwm_value) &&
        safe_pwm_range(motor_pwm_value) &&
        safe_pwm_range(button_pwm_value) &&
        button_prev_interrupt_time >= 0 &&
        servo_prev_interrupt_time >= 0 &&
        motor_prev_interrupt_time >= 0 &&
        state_transition_time_ms >= 0 && 
        safe_percent_range(caffe_servo_percent) &&
        safe_percent_range(caffe_motor_percent) &&
        safe_percent_range(servo_percent) &&
        safe_percent_range(motor_percent) &&
        safe_pwm_range(caffe_servo_pwm_value) &&
        safe_pwm_range(caffe_motor_pwm_value) &&
        caffe_last_int_read_time >= 0 &&
        caffe_mode >= -3 && caffe_mode <= 9 &&
        state >= -1 && state <= 100 &&
        previous_state >= -1 && previous_state <= 100
        
    ) return(1);
    else {
        if (state != STATE_ERROR) {
            // On first entering error state, attempt to null steering and motor
            servo_pwm_value = servo_null_pwm_value;
            motor_pwm_value = motor_null_pwm_value;
            servo.writeMicroseconds(servo_null_pwm_value);
            motor.writeMicroseconds(motor_null_pwm_value);
        }
        state = STATE_ERROR;
        return(0);
    }
}
int safe_pwm_range(int p) {
    if (p < SERVO_MIN) return 0;
    if (p > SERVO_MAX) return 0;
    return(1);
}
int safe_percent_range(int p) {
    if (p > 99) return 0;
    if (p < 0) return 0;
    return(1);
}
//
//////////////////////////////////////////////////////////////////////
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
    Serial.print("(-3,");
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
    Serial.print("(-4,");
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
    Serial.print('(-5,)');
    Serial.println(rate_1);
    Serial.println(')');
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



