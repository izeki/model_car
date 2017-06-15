import rospy
import std_msgs.msg

class Car:

    def __init__(self):
        ### subscribers
        self.state = None
        self.steer = None
        self.motor = None
        self.state_sub = rospy.Subscriber('state', std_msgs.msg.Int32,
                                          callback=self._save_callback,
                                          callback_args=('state',))
        self.steer_sub = rospy.Subscriber('steer', std_msgs.msg.Int32,
                                          callback=self._save_callback,
                                          callback_args=('steer',))
        self.motor_sub = rospy.Subscriber('motor', std_msgs.msg.Int32,
                                          callback=self._save_callback,
                                          callback_args=('motor',))

        ### publishers
        self.steer_cmd_pub = rospy.Publisher('cmd/steer', std_msgs.msg.Int32, queue_size=100)
        self.motor_cmd_pub = rospy.Publisher('cmd/motor', std_msgs.msg.Int32, queue_size=100)

    ###############
    ### Getters ###
    ###############
    
    def get_state(self):
        return self.state
        
    def get_steer(self):
        return self.steer
        
    def get_motor(self):
        return self.motor
        
    ###############
    ### Setters ###
    ###############
    
    def set_steer(self, steer):
        self.steer_cmd_pub.publish(std_msgs.msg.Int32(steer))
        
    def set_motor(self, motor):
        self.motor_cmd_pub.publish(std_msgs.msg.Int32(motor))

    #################
    ### Callbacks ###
    #################

    def _save_callback(self, msg, args):
        attr_name = args[0]
        setattr(self, attr_name, msg.data)


