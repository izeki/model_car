import rospy

import sensor_msgs.msg
import cv_bridge
from cv_bridge import CvBridge, CvBridgeError

class ROSListener:
    def __init__(self, topic, msg_type):
        self.sub = rospy.Subscriber(topic, msg_type, callback=self._callback)
        self.msg = None
        
    def _callback(self, msg):
        self.msg = msg

    def get_msg(self):
        return self.msg

class ImageROSListener:
    def __init__(self, topic, msg_type=sensor_msgs.msg.Image):
        self.ros_listener = ROSListener(topic, msg_type)
        self.bridge = cv_bridge.CvBridge()

    def get_image(self):
        msg = self.ros_listener.get_msg()

        if msg is None:
            return None

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except cv_bridge.CvBridgeError as e:
            return None

        return cv_image
           
