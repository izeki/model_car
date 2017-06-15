#!/usr/bin/env python

import rospy

from bair_car.arduino import Arduino

if __name__ == '__main__':
    rospy.init_node('run_arduino', anonymous=True)
    ard = Arduino()
    rospy.spin()
    
