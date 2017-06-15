import os, serial, threading, Queue
import threading

import rospy
import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
from kzpy3.utils import *

os.environ["ROS_MASTER_URI"] = "http://192.168.43.196:11311"

class External_Arduino:

    STATE_HEADING   = "hdg"
    SENSOR_STATES = (STATE_HEADING)

    def __init__(self, baudrate=115200, timeout=0.25):

        self.ser_sensors = self._setup_serial(baudrate, timeout)
        assert(self.ser_sensors is not None)

        self.state_pub = rospy.Publisher('/bair_car/camera_heading', std_msgs.msg.Float32, queue_size=100)
        rospy.init_node('talker',anonymous=True)

        print('Starting threads')

        threading.Thread(target=self._ros_sensors_thread).start()
    
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
        ser_sensors = None
        for ser in sers:
            for _ in xrange(100):
                try:
                    ser_str = ser.readline()
                    #print ser_str
                    exec('ser_tuple = list({0})'.format(ser_str))
                    #print d2s('ser_tuple = ',ser_tuple)
                    if ser_tuple[0] in External_Arduino.SENSOR_STATES:
                        print('Port {0} is the sensors'.format(ser.port))
                        ser_sensors = ser
                        break
                except:
                    pass
            else:
                print('Unable to identify port {0}'.format(ser.port))
        
        return ser_sensors




    def _ros_sensors_thread(self):
        """
        Receives message from sensors serial and publishes to ros
        """
        info = dict()
        
        while not rospy.is_shutdown():
            try:
            
                sensors_str = self.ser_sensors.readline()
                exec('sensors_tuple = list({0})'.format(sensors_str))

                if sensors_tuple[0] == External_Arduino.STATE_HEADING:
                    assert(len(sensors_tuple) == 2)
                    self.state_pub.publish(std_msgs.msg.Float32(sensors_tuple[1]))
                    #print "published"

            except Exception as e:
                print e
                pass

            


