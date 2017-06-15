import os

import rospkg, rosbag
rospack = rospkg.RosPack()

def parse_rosbag():
    ### open bag file
    bag_path = os.path.join(rospack.get_path('bair_car'), 'rosbags/test.bag')
    bag = rosbag.Bag(bag_path, 'r')
    
    ### read bag file
    bag_messages = bag.read_messages()
    
    ### example of how to parse
    # for topic, msg, t in bag.read_messages():
    #    pass # can do whatever you want here
    
    # import IPython; IPython.embed()

if __name__ == '__main__':
    parse_rosbag()

