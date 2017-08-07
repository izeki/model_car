// Test code for Adafruit GPS modules using MTK3329/MTK3339 driver
//
// This code shows how to listen to the GPS module in an interrupt
// which allows the program to have more 'freedom' - just parse
// when a new NMEA sentence is available! Then access data when
// desired.
//
// Tested and works great with the Adafruit Ultimate GPS module
// using MTK33x9 chipset
//        ------> http://www.adafruit.com/products/746
// Pick one up today at the Adafruit electronics shop 
// and help support open source hardware & software! -ada

#include <Adafruit_GPS.h>
#include <SoftwareSerial.h>

// If you're using a GPS module:
// Connect the GPS Power pin to 5V
// Connect the GPS Ground pin to ground
// If using software serial (sketch example default):
//     Connect the GPS TX (transmit) pin to Digital 3
//     Connect the GPS RX (receive) pin to Digital 2
// If using hardware serial (e.g. Arduino Mega):
//     Connect the GPS TX (transmit) pin to Arduino RX1, RX2 or RX3
//     Connect the GPS RX (receive) pin to matching TX1, TX2 or TX3

// If you're using the Adafruit GPS shield, change 
// SoftwareSerial mySerial(3, 2); -> SoftwareSerial mySerial(8, 7);
// and make sure the switch is set to SoftSerial

// If using software serial, keep this line enabled
// (you can change the pin numbers to match your wiring):
SoftwareSerial mySerial(3, 2);

// If using hardware serial (e.g. Arduino Mega), comment out the
// above SoftwareSerial line, and enable this line instead
// (you can change the Serial number to match your wiring):

//HardwareSerial mySerial = Serial1;


Adafruit_GPS GPS(&mySerial);


// Set GPSECHO to 'false' to turn off echoing the GPS data to the Serial console
// Set to 'true' if you want to debug and listen to the raw GPS sentences. 
#define GPSECHO    false

// this keeps track of whether we're using the interrupt
// off by default!
boolean usingInterrupt = false;
void useInterrupt(boolean); // Func prototype keeps Arduino 0023 happy

void setup()    
{
        
    // connect at 115200 so we can read the GPS fast enough and echo without dropping chars
    // also spit it out
    Serial.begin(115200);
    //Serial.println("Adafruit GPS library basic test!");

    // 9600 NMEA is the default baud rate for Adafruit MTK GPS's- some use 4800
    GPS.begin(9600);
    
    // uncomment this line to turn on RMC (recommended minimum) and GGA (fix data) including altitude
    GPS.sendCommand(PMTK_SET_NMEA_OUTPUT_RMCGGA);
    // uncomment this line to turn on only the "minimum recommended" data
    //GPS.sendCommand(PMTK_SET_NMEA_OUTPUT_RMCONLY);
    // For parsing data, we don't suggest using anything but either RMC only or RMC+GGA since
    // the parser doesn't care about other sentences at this time
    
    // Set the update rate
    GPS.sendCommand(PMTK_SET_NMEA_UPDATE_1HZ);     // 1 Hz update rate
    // For the parsing code to work nicely and have time to sort thru the data, and
    // print it out we don't suggest using anything higher than 1 Hz

    // Request updates on antenna status, comment out to keep quiet
    GPS.sendCommand(PGCMD_ANTENNA);

    // the nice thing about this code is you can have a timer0 interrupt go off
    // every 1 millisecond, and read data from the GPS for you. that makes the
    // loop code a heck of a lot easier!
    useInterrupt(true);

    delay(1000);
    // Ask for firmware version
    mySerial.println(PMTK_Q_RELEASE);
    LED_setup();
}


// Interrupt is called once a millisecond, looks for any new GPS data, and stores it
SIGNAL(TIMER0_COMPA_vect) {
    char c = GPS.read();
    // if you want to debug, this is a good time to do it!
#ifdef UDR0
    if (GPSECHO)
        if (c) UDR0 = c;    
        // writing direct to UDR0 is much much faster than Serial.print 
        // but only one character can be written at a time. 
#endif
}

void useInterrupt(boolean v) {
    if (v) {
        // Timer0 is already used for millis() - we'll just interrupt somewhere
        // in the middle and call the "Compare A" function above
        OCR0A = 0xAF;
        TIMSK0 |= _BV(OCIE0A);
        usingInterrupt = true;
    } else {
        // do not call the interrupt function COMPA anymore
        TIMSK0 &= ~_BV(OCIE0A);
        usingInterrupt = false;
    }
}

uint32_t timer = millis();
void loop()                                         // run over and over again
{
    // in case you are not using the interrupt above, you'll
    // need to 'hand query' the GPS, not suggested :(
    if (! usingInterrupt) {
        // read data from the GPS in the 'main loop'
        char c = GPS.read();
        // if you want to debug, this is a good time to do it!
        if (GPSECHO)
            if (c) Serial.print(c);
    }
    
    // if a sentence is received, we can check the checksum, parse it...
    if (GPS.newNMEAreceived()) {
        // a tricky thing here is if we print the NMEA sentence, or data
        // we end up not listening and catching other sentences! 
        // so be very wary if using OUTPUT_ALLDATA and trytng to print out data
        //Serial.println(GPS.lastNMEA());     // this also sets the newNMEAreceived() flag to false
    
        if (!GPS.parse(GPS.lastNMEA()))     // this also sets the newNMEAreceived() flag to false
            return;    // we can fail to parse a sentence in which case we should just wait for another
    }

    // if millis() or timer wraps around, we'll just reset it
    if (timer > millis())    timer = millis();

    // approximately every 2 seconds or so, print out the current stats
    if (millis() - timer > 500) { 
        timer = millis(); // reset the timer
        /*
        Serial.print("\nTime: ");
        Serial.print(GPS.hour, DEC); Serial.print(':');
        Serial.print(GPS.minute, DEC); Serial.print(':');
        Serial.print(GPS.seconds, DEC); Serial.print('.');
        Serial.println(GPS.milliseconds);
        Serial.print("Date: ");
        Serial.print(GPS.day, DEC); Serial.print('/');
        Serial.print(GPS.month, DEC); Serial.print("/20");
        Serial.println(GPS.year, DEC);
        Serial.print("Fix: "); Serial.print((int)GPS.fix);
        Serial.print(" quality: "); Serial.println((int)GPS.fixquality); 
        if (GPS.fix) {
            Serial.print("Location: ");
            Serial.print(GPS.latitude, 4); Serial.print(GPS.lat);
            Serial.print(", "); 
            Serial.print(GPS.longitude, 4); Serial.println(GPS.lon);
            Serial.print("Location (in degrees, works with Google Maps): ");
            Serial.print(GPS.latitudeDegrees, 4);
            Serial.print(", "); 
            Serial.println(GPS.longitudeDegrees, 4);
            
            Serial.print("Speed (knots): "); Serial.println(GPS.speed);
            Serial.print("Angle: "); Serial.println(GPS.angle);
            Serial.print("Altitude: "); Serial.println(GPS.altitude);
            Serial.print("Satellites: "); Serial.println((int)GPS.satellites);
            */



 
        Serial.print("('GPS2',");
        Serial.print(GPS.hour, DEC);
        Serial.print(',');
        Serial.print(GPS.minute, DEC);
        Serial.print(',');
        Serial.print(GPS.seconds, DEC);
        Serial.print(',');
        //Serial.print(GPS.milliseconds);
        //Serial.print(",");
        Serial.print(GPS.day, DEC);
        Serial.print(',');
        Serial.print(GPS.month, DEC);
        Serial.print(",");
        Serial.print(GPS.year, DEC);
        Serial.print(",");
        Serial.print((int)GPS.fix);
        Serial.print(",");
        Serial.print((int)GPS.fixquality); 
        if (1) { //(GPS.fix) {
            Serial.print(",");
            //Serial.print(GPS.latitude, 4); Serial.print(GPS.lat);
            //Serial.print(", "); 
            //Serial.print(GPS.longitude, 4); Serial.println(GPS.lon);
            //Serial.print(",");
            Serial.print(GPS.latitudeDegrees, 5);
            Serial.print(","); 
            Serial.print(GPS.longitudeDegrees, 5);
            Serial.print(",");
            Serial.print(GPS.speed);
            Serial.print(",");
            Serial.print(GPS.angle);
            Serial.print(",");
            Serial.print(GPS.altitude);
            Serial.print(",");
            Serial.print((int)GPS.satellites);
        Serial.println(")");
        }
    }
    LED_loop();
}



///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#include <Wire.h>
#include "Adafruit_LEDBackpack.h"
#include "Adafruit_GFX.h"

Adafruit_BicolorMatrix matrix = Adafruit_BicolorMatrix();

const int led_LEFT =    2;
const int led_RIGHT = 11;
const int led_DATA =    3;
const int led_STATE_1 = 8;
const int led_STATE_2 = 9;
const int led_STATE_3 = 10;
const int led_STATE_4 = 12;
/*
const int button_A =    5;
const int button_B =    6;
const int button_C =    7;
const int button_D =    4;

int button_A_state = LOW;
int button_B_state = LOW;
int button_C_state = LOW;
int button_D_state = LOW;
*/

static const uint8_t PROGMEM
    left_bmp[] =
    {   B10000000,
        B10000000,
        B10000000,
        B10000000,
        B10000000,
        B10000000,
        B10000000,
        B10000000 },
    right_bmp[] =
    {   B00000001,
        B00000001,
        B00000001,
        B00000001,
        B00000001,
        B00000001,
        B00000001,
        B00000001 },    
    one_bmp[] =
    {   B00000000,
        B00011000,
        B00111000,
        B00011000,
        B00011000,
        B00011000,
        B00111100,
        B00000000 },
     two_bmp[] =
    {   B00000000,
        B00011110,
        B00110110,
        B01101100,
        B00011000,
        B00111110,
        B01111110,
        B00000000 },
        three_bmp[] =
    {   B00000000,
        B00111100,
        B01100110,
        B00011100,
        B00001110,
        B01100110,
        B00111100,
        B00000000 },
        four_bmp[] =
    {   B00000000,
        B00001110,
        B00011110,
        B00110110,
        B01111110,
        B00000110,
        B00000110,
        B00000000 },
        five_bmp[] =
    {   B00000000,
        B01111110,
        B01100000,
        B01100000,
        B01111110,
        B00000110,
        B01111110,
        B00000000 },
        six_bmp[] =
    {   B00000000,
        B00111000,
        B01100000,
        B01100000,
        B01111110,
        B01100110,
        B01111110,
        B00000000 },
        seven_bmp[] =
    {   B00000000,
        B01111110,
        B01100110,
        B00001100,
        B00011000,
        B00011000,
        B00011000,
        B00000000 },
    saving_data_bmp[] =
    {   B01111110,
        B01100000,
        B00110000,
        B00011000,
        B00001100,
        B00000110,
        B00000110,
        B01111110 };

void left_turn() {
    matrix.drawBitmap(0, 0, left_bmp, 8, 8, LED_RED);
    matrix.writeDisplay();
}
void right_turn() {
    matrix.drawBitmap(0, 0, right_bmp, 8, 8, LED_RED);
    matrix.writeDisplay();
}
void one() {
    //matrix.clear();
    matrix.drawBitmap(0, 0, one_bmp, 8, 8, LED_GREEN);
    matrix.writeDisplay();
    matrix.blinkRate(0);
}
void two() {
    //matrix.clear();
    matrix.drawBitmap(0, 0, two_bmp, 8, 8, LED_RED);
    matrix.writeDisplay();
    matrix.blinkRate(0);
}
void three() {
    //matrix.clear();
    matrix.drawBitmap(0, 0, three_bmp, 8, 8, LED_YELLOW);
    matrix.writeDisplay();
    matrix.blinkRate(0);
}
void four() {
    //matrix.clear();
    matrix.drawBitmap(0, 0, four_bmp, 8, 8, LED_RED);
    matrix.writeDisplay();
    matrix.blinkRate(1);

}
void five() {
    //matrix.clear();
    matrix.drawBitmap(0, 0, five_bmp, 8, 8, LED_YELLOW);
    matrix.writeDisplay();
    matrix.blinkRate(0);
}
void six() {
    //matrix.clear();
    matrix.drawBitmap(0, 0, six_bmp, 8, 8, LED_YELLOW);
    matrix.writeDisplay();
    matrix.blinkRate(0);
}
void seven() {
    //matrix.clear();
    matrix.drawBitmap(0, 0, seven_bmp, 8, 8, LED_YELLOW);
    matrix.writeDisplay();
    matrix.blinkRate(0);
}
void save_data() {
    matrix.drawBitmap(0, 0, saving_data_bmp, 8, 8, LED_YELLOW);
    matrix.writeDisplay();
    matrix.blinkRate(1);
}

void clear_matrix() {
    matrix.clear();
}

void LED_setup() {

    matrix.begin(0x70);    // pass in the address
    matrix.setRotation(3);
    matrix.blinkRate(0);
}
int sig_data = 0;
int sig_state = 0;
int sig_left_right = 0;
int a = 0;
int b = 0;
int c = 0;
int d = 0;
int e = 0;

void LED_loop() {

    int parsed_int = Serial.parseInt();

    
    int I = 0;
    
    if (parsed_int > 0) {
        I = parsed_int;
    }
/*
    button_A_state = digitalRead(button_A);
    button_B_state = digitalRead(button_B);
    button_C_state = digitalRead(button_C);
    button_D_state = digitalRead(button_D);
*/
    /*
    Serial.println(button_A_state);
    Serial.println(button_B_state);
    Serial.println(button_C_state);
    Serial.println(button_D_state);
    */
    
    if (I > 0) {
            a = I/10000;
            b = (I-a*10000)/1000;
            c = (I-a*10000-b*1000)/100;
            d = (I-a*10000-b*1000-c*100)/10;
            e = (I-a*10000-b*1000-c*100-d*10);
            sig_state = d;
            sig_data = e;
    }

    matrix.clear();
    /*
    if (button_A_state == HIGH) {
        left_turn();
        //digitalWrite(led_LEFT, HIGH);
        //Serial.println("('left_right',-1)");
    }
    else if (button_B_state == HIGH) {
        right_turn();
        //digitalWrite(led_RIGHT, HIGH);
        //Serial.println("('left_right',1)");
    }
    else {

        //Serial.println("('left_right',0)");
    }
    if (button_A_state == LOW) {
        digitalWrite(led_LEFT, LOW);
    }
    if (button_B_state == LOW) {
        digitalWrite(led_RIGHT, LOW);
    }
*/
    
    if (sig_data == 1) {
        save_data();
        //digitalWrite(led_DATA, HIGH);
    }
    else if (sig_data == 2) {
        left_turn();
        //digitalWrite(led_DATA, HIGH);
    }
    else if (sig_data == 3) {
        right_turn();
        //digitalWrite(led_DATA, HIGH);
    }
    if (sig_state == 4) {
        four();
        /*
        digitalWrite(led_STATE_1, HIGH);
        digitalWrite(led_STATE_2, HIGH);
        digitalWrite(led_STATE_3, HIGH);
        digitalWrite(led_STATE_4, HIGH);
        */
    }
    else if (sig_state == 3) {
        three();
        /*
        digitalWrite(led_STATE_1, HIGH);
        digitalWrite(led_STATE_2, HIGH);
        digitalWrite(led_STATE_3, HIGH);
        digitalWrite(led_STATE_4, LOW);
        */
    }/*
    else if (sig_state > 4) {
        three();
        /*
        digitalWrite(led_STATE_1, HIGH);
        digitalWrite(led_STATE_2, HIGH);
        digitalWrite(led_STATE_3, HIGH);
        digitalWrite(led_STATE_4, LOW);
        /
    }*/
    else if (sig_state == 5) {
        five();
        /*
        digitalWrite(led_STATE_1, HIGH);
        digitalWrite(led_STATE_2, HIGH);
        digitalWrite(led_STATE_3, HIGH);
        digitalWrite(led_STATE_4, LOW);
        */
    }
    else if (sig_state == 6) {
        six();
        /*
        digitalWrite(led_STATE_1, HIGH);
        digitalWrite(led_STATE_2, HIGH);
        digitalWrite(led_STATE_3, HIGH);
        digitalWrite(led_STATE_4, LOW);
        */
    }
    else if (sig_state == 7) {
        seven();
        /*
        digitalWrite(led_STATE_1, HIGH);
        digitalWrite(led_STATE_2, HIGH);
        digitalWrite(led_STATE_3, HIGH);
        digitalWrite(led_STATE_4, LOW);
        */
    }/*/
    else if (sig_state == 2) {
        two();
        /*
        digitalWrite(led_STATE_1, HIGH);
        digitalWrite(led_STATE_2, HIGH);
        digitalWrite(led_STATE_3, LOW);
        digitalWrite(led_STATE_4, LOW);
        */
    }
    else if (sig_state == 1) {
        one();
        /*
        digitalWrite(led_STATE_1, HIGH);
        digitalWrite(led_STATE_2, LOW);
        digitalWrite(led_STATE_3, LOW);
        digitalWrite(led_STATE_4, LOW);
        */
    }
/*
        Serial.print("(");
        Serial.print(I);
        Serial.print(",");        
        Serial.print(a);
        Serial.print(",");
        Serial.print(b);
        Serial.print(",");
        Serial.print(sig_state);
        Serial.print(",");
        Serial.print(sig_left_right);
        Serial.print(",");
         Serial.print(sig_data);
        Serial.print(button_A_state);
        Serial.print(",");
        Serial.print(button_B_state);        
        Serial.println(")");
*/     
    delay(10);
}

