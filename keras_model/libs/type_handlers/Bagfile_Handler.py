'''    
Created on Apr 11, 2017

@author: picard
'''
import sys
import os
import rosbag
from cv_bridge import CvBridge

class Bagfile_Handler(object):
    
    print("Reading Bagfile")
    bag = None
    bag_access = None
    bridge = CvBridge()
    
    # pickle_file = None
    data_for_pickle_file = []
    old_evasion_data = []
    
    def __init__(self, bag_filepath, topic_list):
        self.bag = rosbag.Bag(bag_filepath)
        head, tail = os.path.split(bag_filepath)
        self.bag_access = self.bag.read_messages(topics=topic_list).__iter__()
        
    def __del__(self):
        try:
            self.bag.close()
        except Exception as ex:
            pass
       
    def get_bag_content(self):
        try:
            topic, msg, timestamp = self.bag_access.next()
        except StopIteration:
            return None, None, None
        return topic, msg, timestamp               
