/////////////////////
// All possible states (i.e. first numbers sent in tuples along serial)
//
// These are the possible states of the control system.
// States are reached by button presses or drive commands, except for error state.
#define STATE_HUMAN_FULL_CONTROL            1
#define STATE_LOCK                          2
#define STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR 3
#define STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR 5
#define STATE_CAFFE_CAFFE_STEER_CAFFE_MOTOR 6
#define STATE_CAFFE_HUMAN_STEER_CAFFE_MOTOR 7
#define STATE_LOCK_CALIBRATE                4
#define STATE_ERROR                         -1
// Now for the sensors
#define STATE_GPS                           "'gps'"
#define STATE_GYRO                          "'gyro'"
#define STATE_SONAR                         "'sonar'"
#define STATE_ENCODER                       "'encoder'"
//
/////////////////////


/////////////////////
// Servo constants
//
// These define extreme min an max values that should never be broken.
#define SERVO_MAX   2000
#define MOTOR_MAX   SERVO_MAX
#define BUTTON_MAX  SERVO_MAX
#define SERVO_MIN   500
#define MOTOR_MIN   SERVO_MIN
#define BUTTON_MIN  SERVO_MIN
//
/////////////////////


/////////////////////
// The hand-held radio controller has two buttons. Pressing the upper or lower
// allows for reaching separate PWM levels: ~ 1710, 1200, 1000, and 888
// These are used for different control states.
#define BUTTON_A 1710 // Human in full control of driving
#define BUTTON_B 1200 // Lock state
#define BUTTON_C 964  // Caffe steering, human on accelerator
#define BUTTON_D 850  // Calibration of steering and motor control ranges
#define BUTTON_DELTA 50 // range around button value that is considered in that value
//
/////////////////////


