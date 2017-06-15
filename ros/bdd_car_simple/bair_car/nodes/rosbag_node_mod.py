#!/usr/bin/env python
from kzpy3.utils import *
import os, sys, shutil, subprocess, time
import rospy
import std_msgs.msg


from computer_name import foldername


if __name__ == '__main__':
    rospy.init_node('rosbag_node', anonymous=True)
    save_pub = rospy.Publisher('signals', std_msgs.msg.Int32, queue_size=100)
    

    fl = gg('/home/ubuntu/catkin_ws/src/bair_car/rosbags/*')

    for f in fl:
         os.remove(f)

    assert(len(sys.argv) >= 3)

    bag_rec_folder = sys.argv[1] # '/home/ubuntu/catkin_ws/src/bair_car/rosbags'
    bag_mv_folder = sys.argv[2] # '/media/ubuntu/3131-3031/rosbags'
    bag_mv_folder = opj(bag_mv_folder,foldername)

    unix('mkdir '+bag_mv_folder)
    unix('mkdir  '+opj(bag_mv_folder,'.caf'))
    unix('mkdir  '+opj(bag_mv_folder,'.bair_car'))

    unix('scp -r /home/ubuntu/catkin_ws/src/bair_car ' + opj(bag_mv_folder,'.bair_car'))
    unix('scp /home/ubuntu/Desktop/z2/z2.caffemodel ' + opj(bag_mv_folder,'.caf'))
    
    assert(os.path.exists(bag_rec_folder))
    assert(os.path.exists(bag_mv_folder))

    rate = rospy.Rate(2.0)
    while not rospy.is_shutdown():
	save_pub.publish(std_msgs.msg.Int32(1))
        for f in os.listdir(bag_rec_folder):
            if '.bag' != os.path.splitext(f)[1]:
                continue
            save_pub.publish(std_msgs.msg.Int32(2) )
            print('Moving {0}'.format(f))
            f_rec = os.path.join(bag_rec_folder, f)
            f_mv = os.path.join(bag_mv_folder, f)
            # shutil.copy(f_rec, f_mv)
            start = time.time()
            subprocess.call(['mv', f_rec, f_mv])
            elapsed = time.time() - start
            print('Done in {0} secs\n'.format(elapsed))
            save_pub.publish(std_msgs.msg.Int32(1))
            
        rate.sleep()

