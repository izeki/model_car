#!/usr/bin/env python
from model_car.utils import *
import os, sys, shutil, subprocess, signal
import rospy
import std_msgs.msg


from model_car.car_run_params import foldername

time.sleep(3)

def terminate_process_and_children(p):
    import psutil
    process = psutil.Process(p.pid)
    for sub_process in process.children(recursive=True):
        sub_process.send_signal(signal.SIGINT)
    p.wait()  # we wait for children to terminate
    p.terminate()

if __name__ == '__main__':
    rospy.init_node('rosbag_record_node', anonymous=True)
    sig_pub = rospy.Publisher('signals', std_msgs.msg.Int32, queue_size=100)
    

    assert(len(sys.argv) >= 3)
    #print('argv:{}'.format(sys.argv))
    car_name = sys.argv[1]
    bag_base_folder = sys.argv[2]  # "/media/ubuntu/rosbags"
    rosbag_args = ''.join(' ' + str(e) for e in sys.argv[3:-3]) # 
    bag_rec_folder = opj(bag_base_folder,foldername)

    unix('mkdir '+bag_rec_folder)
    unix('mkdir  '+opj(bag_rec_folder,'.AI'))
    unix('mkdir  '+opj(bag_rec_folder,'.bair_car'))

    unix('scp -r /home/ubuntu/catkin_ws/src/bair_car ' + opj(bag_rec_folder,'.bair_car'))
    #unix('scp /home/ubuntu/model_car/model_car/model/z2_color_tf.npy ' + opj(bag_rec_folder,'.AI'))
    unix('scp /home/ubuntu/model_car/model_car/model/z2_color_version_1b_final_run56652.hdf5' + opj(bag_rec_folder,'.AI'))
    
    assert(os.path.exists(bag_rec_folder))
    
    # start rosbag record process
    print('rosbag record --split --size=1024 -b 2048 --lz4 -o {} {}'.format(car_name, rosbag_args))
    rosbag_process = subprocess.Popen('rosbag record --split --size=1024 -b 2048 --lz4 -o {} {}'.format(car_name, rosbag_args), stdin=subprocess.PIPE, shell=True, cwd=bag_rec_folder)

    rate = rospy.Rate(1)    
    
    while not rospy.is_shutdown():
        sig_pub.publish(std_msgs.msg.Int32(1))            
        rate.sleep()
    print('Terminate rosbag record process...\n')
    terminate_process_and_children(rosbag_process)
    print('rosbag record process terminated.\n')

