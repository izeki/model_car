#!/usr/bin/env python


"""
from kzpy3.utils import *
import rospy
########################################################
#          ROSPY SETUP SECTION
import roslib
import std_msgs.msg

rospy.init_node('signal_node',anonymous=True)
#
#signal = -0
signal = 0
def signal_callback(data):
    global signal
    signal = data.data

##
########################################################


rospy.Subscriber('/signals', std_msgs.msg.Int32,signal_callback)

while not rospy.is_shutdown():
    try:
        time.sleep(0.2)
        print signal
        pass

    except:
        pass


rospy.spin()
"""