#!/usr/bin/env python
from model_car.utils import *
import os, sys, shutil, subprocess, time
import rospy
import std_msgs.msg


from model_car.car_run_params import foldername

time.sleep(3)

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
    unix('mkdir  '+opj(bag_mv_folder,'.AI'))
    unix('mkdir  '+opj(bag_mv_folder,'.bair_car'))

    unix('scp -r /home/ubuntu/catkin_ws/src/bair_car ' + opj(bag_mv_folder,'.bair_car'))
    unix('scp /home/ubuntu/model_car/model_car/model/z2_color_tf.npy ' + opj(bag_mv_folder,'.AI'))
    
    assert(os.path.exists(bag_rec_folder))
    assert(os.path.exists(bag_mv_folder))

    rate = rospy.Rate(1.0)
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
            #unix('rm '+opj(bag_rec_folder,'*.bag')) # 27 Nov 2016, to remove untransferred bags
            print('Done in {0} secs\n'.format(elapsed))
            save_pub.publish(std_msgs.msg.Int32(1))
            
        rate.sleep()

