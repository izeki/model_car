import os, serial, threading, Queue
import threading

import rospy
import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
from model_car.utils import *
#
class Arduino:

    STATE_HUMAN_FULL_CONTROL            = 1
    STATE_LOCK                          = 2
    STATE_AI_AI_STEER_HUMAN_MOTOR       = 3
    STATE_AI_HUMAN_STEER_HUMAN_MOTOR    = 5
    STATE_AI_AI_STEER_AI_MOTOR          = 6
    STATE_AI_HUMAN_STEER_AI_MOTOR       = 7
    STATE_LOCK_CALIBRATE                = 4
    STATE_ERROR                         = -1
    CONTROL_STATES = (STATE_HUMAN_FULL_CONTROL,
                      STATE_LOCK,
                      STATE_AI_AI_STEER_HUMAN_MOTOR,
                      STATE_AI_AI_STEER_AI_MOTOR,
                      STATE_AI_HUMAN_STEER_HUMAN_MOTOR,
                      STATE_AI_HUMAN_STEER_AI_MOTOR,
                      STATE_LOCK_CALIBRATE,
                      STATE_ERROR)
    
    #STATE_GPS                           = "gps"
    STATE_GPS                           = "GPS2"
    STATE_GYRO                          = "gyro"
    STATE_GYRO_HEADING                  = "head"
    STATE_ACC                           = "acc"
    SENSOR_STATES = (STATE_GYRO,
                    STATE_GYRO_HEADING,
                    STATE_ACC)
    SIGNALS = (STATE_GPS)

    def __init__(self, baudrate=115200, timeout=0.25):
        ### setup serial ports
        self.ser_servos, self.ser_sensors, self.ser_signals = self._setup_serial(baudrate, timeout)
        assert(self.ser_servos is not None)
        if self.ser_sensors is None:
            print("****** self.ser_sensors is None ******")
        if self.ser_signals is None:
            print("****** self.ser_signals is None ******")
        #assert(self.ser_sensors is not None)
        #assert(self.ser_signals is not None)
        
        ### control publishers (from Arduino)
        self.state_pub = rospy.Publisher('state', std_msgs.msg.Int32, queue_size=100)
        self.steer_pub = rospy.Publisher('steer', std_msgs.msg.Int32, queue_size=100)
        self.motor_pub = rospy.Publisher('motor', std_msgs.msg.Int32, queue_size=100)
        self.encoder_pub = rospy.Publisher('encoder', std_msgs.msg.Float32, queue_size=100)
        self.state_transition_time_s_pub = rospy.Publisher('state_transition_time_s', std_msgs.msg.Int32, queue_size=100)
        ### sensor publishers (from Arduino)

        #self.gps_pub = rospy.Publisher('gps', sensor_msgs.msg.NavSatFix, queue_size=100)

        self.GPS2_hour_pub = rospy.Publisher('GPS2_hour', std_msgs.msg.Int32, queue_size=100)
        self.GPS2_min_pub = rospy.Publisher('GPS2_min', std_msgs.msg.Int32, queue_size=100)
        self.GPS2_sec_pub = rospy.Publisher('GPS2_sec', std_msgs.msg.Int32, queue_size=100)
        self.GPS2_day_pub = rospy.Publisher('GPS2_day', std_msgs.msg.Int32, queue_size=100)
        self.GPS2_mon_pub = rospy.Publisher('GPS2_mon', std_msgs.msg.Int32, queue_size=100)
        self.GPS2_yr_pub = rospy.Publisher('GPS2_yr', std_msgs.msg.Int32, queue_size=100)
        self.GPS2_fix_pub = rospy.Publisher('GPS2_fix', std_msgs.msg.Int32, queue_size=100)
        self.GPS2_qual_pub = rospy.Publisher('GPS2_qual', std_msgs.msg.Int32, queue_size=100)
        self.GPS2_lat_pub = rospy.Publisher('GPS2_lat', std_msgs.msg.Float32, queue_size=100)
        self.GPS2_long_pub = rospy.Publisher('GPS2_long', std_msgs.msg.Float32, queue_size=100)
        self.GPS2_speed_pub = rospy.Publisher('GPS2_speed', std_msgs.msg.Float32, queue_size=100)
        self.GPS2_angle_pub = rospy.Publisher('GPS2_angle', std_msgs.msg.Float32, queue_size=100)
        self.GPS2_alt_pub = rospy.Publisher('GPS2_alt', std_msgs.msg.Float32, queue_size=100)
        self.GPS2_sat_pub = rospy.Publisher('GPS2_sat', std_msgs.msg.Int32, queue_size=100)

        self.gyro_pub = rospy.Publisher('gyro', geometry_msgs.msg.Vector3, queue_size=100)
        self.gyro_heading_pub = rospy.Publisher('gyro_heading', geometry_msgs.msg.Vector3, queue_size=100)
        self.acc_pub = rospy.Publisher('acc', geometry_msgs.msg.Vector3, queue_size=100)
        self.sonar_pub = rospy.Publisher('sonar', std_msgs.msg.Int32, queue_size=100)
        self.signals_pub = rospy.Publisher('left_right', std_msgs.msg.Int32, queue_size=100)
        ### subscribers (info sent to Arduino)
        self.cmd_steer_sub = rospy.Subscriber('cmd/steer', std_msgs.msg.Int32,
                                              callback=self._cmd_steer_callback)
        self.cmd_motor_sub = rospy.Subscriber('cmd/motor', std_msgs.msg.Int32,
                                              callback=self._cmd_motor_callback)
        self.cmd_signal_sub = rospy.Subscriber('/signals', std_msgs.msg.Int32,
                                              callback=self._cmd_signal_callback)
        self.cmd_signals_sub = rospy.Subscriber('/left_right', std_msgs.msg.Int32,
                                              callback=self._cmd_signal_callback)
        self.cmd_steer_queue = Queue.Queue()
        self.cmd_motor_queue = Queue.Queue()
        self.cmd_signal_queue = Queue.Queue()

        self.signal = 1
        self.info_state = 0

        ### start background ros thread
        print('Starting threads')

        threading.Thread(target=self._ros_servos_thread).start()
        print('threading.Thread(target=self._ros_servos_thread).start()')

        if self.ser_sensors is not None:
            threading.Thread(target=self._ros_sensors_thread).start()
            print('threading.Thread(target=self._ros_sensors_thread).start()')
            
        if self.ser_signals is not None:
            threading.Thread(target=self._ros_signals_thread).start()
            print('threading.Thread(target=self._ros_signals_thread).start()')
        

    #############
    ### Setup ###
    #############
    
    def _setup_serial(self, baudrate, timeout):
        sers = []
        ACM_ports = [os.path.join('/dev', p) for p in os.listdir('/dev') if 'ttyACM' in p]
        for ACM_port in ACM_ports:
            try:
                sers.append(serial.Serial(ACM_port, baudrate=baudrate, timeout=timeout))
                print('Opened {0}'.format(ACM_port))
            except:
                pass
                
        ### determine which serial port is which
        ser_servos = None
        ser_sensors = None
        ser_signals = None
        for ser in sers:
            for _ in xrange(100):
                try:
                    ser_str = ser.readline()
                    exec('ser_tuple = list({0})'.format(ser_str))
                    if ser_tuple[0] in Arduino.CONTROL_STATES:
                        print('Port {0} is the servos'.format(ser.port))
                        ser_servos = ser
                        break
                    elif ser_tuple[0] in Arduino.SENSOR_STATES:
                        print('Port {0} is the sensors'.format(ser.port))
                        ser_sensors = ser
                        break
                    elif ser_tuple[0] in Arduino.SIGNALS:
                        print('Port {0} is the signals'.format(ser.port))
                        ser_signals = ser
                        break
                except:
                    pass
            else:
                print('Unable to identify port {0}'.format(ser.port))
        
        return ser_servos, ser_sensors, ser_signals

    ###################
    ### ROS methods ###
    ###################

    def _ros_servos_thread(self):
        """
        Sends/receives message from servos serial and
        publishes/subscribes to ros
        """
        info = dict()
        
        while not rospy.is_shutdown():
            try:        
                ### read servos serial
                servos_str = self.ser_servos.readline()
                exec('servos_tuple = list({0})'.format(servos_str))
                ### parse servos serial
                info['state'], info['steer'], info['motor'], info['encoder'], info['state_transition_time_s'] = servos_tuple
                self.info_state = info['state'] # 9/27/16
                ### publish ROS
                self.state_pub.publish(std_msgs.msg.Int32(info['state']))
                self.steer_pub.publish(std_msgs.msg.Int32(info['steer']))
                # examine turning direction
                if info['steer'] > 49: #right turn
                    self.signals_pub.publish(std_msgs.msg.Int32(3))
                elif info['steer'] < 49: #left turn
                    self.signals_pub.publish(std_msgs.msg.Int32(2))
                self.motor_pub.publish(std_msgs.msg.Int32(info['motor']))
                self.encoder_pub.publish(std_msgs.msg.Float32(info['encoder']))
                self.state_transition_time_s_pub.publish(std_msgs.msg.Int32(info['state_transition_time_s']))                
               
                ### write servos serial
                write_to_servos = False
                for var, queue in (('cmd_steer', self.cmd_steer_queue),
                                   ('cmd_motor', self.cmd_motor_queue)):
                    if not queue.empty():
                        write_to_servos = True
                        info[var] = queue.get()
                        #print('Setting {0} to {1}'.format(var, info[var]))
		#info={}
		#info['steer']=30
		#info['motor']=55
		#info['steer'] = STEER
		#write_to_servos = True
                        
                if write_to_servos:
                    servos_write_int = 10000*1 + 100*info['cmd_steer'] + info['cmd_motor']
                    servos_write_str = '( {0} )'.format(servos_write_int)
                    #print(('servos_write_str',servos_write_str))#,STEER))
                    self.ser_servos.write(servos_write_str)
                   


            except Exception as e:
                print e
                # print e
 

    def _ros_signals_thread(self):
        """

        """
        info = dict()
        
        while not rospy.is_shutdown():
            try:        
                # Signal to publish from signals Arduino
                ### read sensors serial
                signals_str = self.ser_signals.readline()
                exec('signals_tuple = list({0})'.format(signals_str))
                #print signals_tuple
                ### parse servos serial and publish to ROS
                
                #print d2s("here 1 ", data[1])
                if signals_tuple[0] == Arduino.STATE_GPS:
                    # lat, long (floats)
                    #print d2s("here 2 ", data[2])
                    assert(len(signals_tuple) == 15)
                    #print d2s("here 3 ", data[3])
                    self.GPS2_hour_pub.publish(std_msgs.msg.Int32(signals_tuple[1]))
                    self.GPS2_min_pub.publish(std_msgs.msg.Int32(signals_tuple[2]))
                    self.GPS2_sec_pub.publish(std_msgs.msg.Int32(signals_tuple[3]))
                    self.GPS2_day_pub.publish(std_msgs.msg.Int32(signals_tuple[4]))
                    self.GPS2_mon_pub.publish(std_msgs.msg.Int32(signals_tuple[5]))
                    self.GPS2_yr_pub.publish(std_msgs.msg.Int32(signals_tuple[6]))
                    self.GPS2_fix_pub.publish(std_msgs.msg.Int32(signals_tuple[7]))
                    self.GPS2_qual_pub.publish(std_msgs.msg.Int32(signals_tuple[8]))
                    self.GPS2_lat_pub.publish(std_msgs.msg.Float32(signals_tuple[9]))
                    self.GPS2_long_pub.publish(std_msgs.msg.Float32(signals_tuple[10]))
                    self.GPS2_speed_pub.publish(std_msgs.msg.Float32(signals_tuple[11]))
                    self.GPS2_angle_pub.publish(std_msgs.msg.Float32(signals_tuple[12]))
                    self.GPS2_alt_pub.publish(std_msgs.msg.Float32(signals_tuple[13]))
                    self.GPS2_sat_pub.publish(std_msgs.msg.Int32(signals_tuple[14]))
                    #gps_msg = sensor_msgs.msg.NavSatFix(longitude=data[0], latitude=data[1])
                    #gps_msg.header.stamp = rospy.Time.now()
                    #self.gps_pub.publish(gps_msg)
                    
                else:
                    pass
                """
                gps_config = [['hour','Int32'],
                    ['min','Int32'],
                    ['sec','Int32'],
                    ['day','Int32'],
                    ['mon','Int32'],
                    ['yr','Int32'],
                    ['fix','Int32'],
                    ['qual','Int32'],
                    ['lat','Float32'],
                    ['long','Float32'],
                    ['speed','Float32'],
                    ['angle','Float32'],
                    ['alt','Float32'],
                    ['sat','Int32']]

                for el in gps_config:
                    name = 'GPS2_'+el[0]
                    typ = el[1]
                    print "self."+name+"_pub = rospy.Publisher('"+name+"', std_msgs.msg."+typ+", queue_size=100)"     

                ctr = 1
                for el in gps_config:
                    name = 'GPS2_'+el[0]
                    typ = el[1]
                    print "self."+name+"_pub.publish(std_msgs.msg."+typ+"(data["+str(ctr)+"]))" 
                    ctr += 1   
                """
                    
                # Signal to send to signals Arduino
                #write_to_signals = False
                write_to_signals = True
                #for var, queue in (('cmd_signal', self.cmd_signal_queue)):
                #    if not queue.empty():
                #        write_to_signals = True
                #        info[var] = queue.get()
                if write_to_signals:                    
                    #signals_ser_int = 10*info['state'] + info['cmd_signal']
                    #signals_ser_str = '( {0}) '.format(signals_ser_int)
                    signals_ser_str = d2n('(',10*self.info_state + self.signal,')')
                    #print signals_ser_str    
                    #print(d2n('(',self.signal,')'))
                    self.ser_signals.write(signals_ser_str)
                    ### print stuff
                    # print servos_tuple                

            except Exception as e:
                pass
                #print '_ros_signals_thread'
                #print signals_str
                #print e
                # print e
 


    def _ros_sensors_thread(self):
        """
        Receives message from sensors serial and publishes to ros
        """
        info = dict()
        
        while not rospy.is_shutdown():
            try:
            
                ### read sensors serial
                sensors_str = self.ser_sensors.readline()

                exec('sensors_tuple = list({0})'.format(sensors_str))
                ### parse servos serial and publish to ROS
                sensor = sensors_tuple[0]
                data = sensors_tuple[1:]
                if sensor == Arduino.STATE_GPS:
                    # lat, long (floats)
                    assert(len(data) == 2)
                    gps_msg = sensor_msgs.msg.NavSatFix(longitude=data[0], latitude=data[1])
                    gps_msg.header.stamp = rospy.Time.now()
                    self.gps_pub.publish(gps_msg)
                elif sensor == Arduino.STATE_GYRO:
                    # x, y, z (floats)
                    assert(len(data) == 3)
                    self.gyro_pub.publish(geometry_msgs.msg.Vector3(*data))
                elif sensor == Arduino.STATE_GYRO_HEADING:
                    # x, y, z (floats)
                    assert(len(data) == 3)
                    self.gyro_heading_pub.publish(geometry_msgs.msg.Vector3(*data))
                elif sensor == Arduino.STATE_ACC:
                    # x, y, z (floats)
                    assert(len(data) == 3)
                    self.acc_pub.publish(geometry_msgs.msg.Vector3(*data))
                elif sensor == Arduino.STATE_SONAR:
                    # dist (int)
                    assert(len(data) == 1)
                    self.sonar_pub.publish(std_msgs.msg.Int32(data[0]))
                else:
                    pass
                
                ### print stuff
                # print sensors_tuple

            except Exception as e:
                pass

    #################
    ### Callbacks ###
    #################
            
    def _cmd_steer_callback(self, msg):
        if msg.data >= 0 and msg.data < 100:
            self.cmd_steer_queue.put(msg.data)
        
    def _cmd_motor_callback(self, msg):
        if msg.data >= 0 and msg.data < 100:
            self.cmd_motor_queue.put(msg.data)

    def _cmd_signal_callback(self, msg):
        self.signal = msg.data
        #print msg.data
        if msg.data >= 0 and msg.data < 100:
            self.cmd_signal_queue.put(msg.data)

