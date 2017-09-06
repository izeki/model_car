# This is used to specifiy caffe mode and data file name information

import os
import numpy as np

from keras_model.libs.utils2 import time_str, opjh, print_stars0, print_stars1


print_stars0();print(__file__);print_stars1()

computer_name = "MR_Unknown"
try:  
   computer_name = os.environ["COMPUTER_NAME"]
except KeyError: 
   print """********** Please set the environment variable computer_name ***********
   e.g.,
   export COMPUTER_NAME="Mr_Orange"
   """


####################### general car settings ################
#
for i in range(5):
	print('*************' + computer_name + '***********')
Direct = 1.
Follow = 0.
Play = 0.
Furtive = 0.
AI = 0.0
Racing = 0.0
Location =  'generic' #Smyth_tape'

#weights_file_path = opjh("model_car/model_car/model/z2_color_tf.npy")
#weights_file_path = opjh('model_car/keras_model/nets/z2_color_version_1b_final.hdf5') #
weights_file_path = opjh('model_car/keras_model/nets/z2_color_squeeze_net.hdf5') #
use_AI = True
verbose = False
#use_caffe = True
n_avg_IMU = 10
NETWORK = 111
I_ROBOT = 222
who_is_in_charge = I_ROBOT
robot_steer = 49
robot_motor = 57
robot_steer_gain = 0.8

past_to_present_proportion = 0.75
steer_momentum = 0.5

X_PARAM = 1.0
Y_PARAM = 1.0
HEADING_DELTA_PARAM = 0.1
STEER_FROM_XY = False
radius = 0.5
#potential_graph_blur = 4
print_marker_ids = False
img_width = 35

steer_gain = 1.0
motor_gain = 1.0
#acc2rd_threshold = 150

PID_min_max = [1.5,2.5]
robot_acc2rd_threshold = 15
robot_acc_y_exit_threshold = 0
potential_acc2rd_collision = 10
potential_motor_freeze_collision = 20
acc_y_tilt_event = 1000

gyro_freeze_threshold = 500
acc_freeze_threshold_x = 12
acc_freeze_threshold_y = 12
acc_freeze_threshold_z = 12
acc_freeze_threshold_z_neg = -7
motor_freeze_threshold = 60



#
###################################################################

####################### specific car settings ################
#
if computer_name == 'Mr_Orange':
    #motor_gain = 1.0
    pass
if computer_name == 'Mr_Silver':
    #motor_gain = 1.0
    pass
if computer_name == 'Mr_Blue':
    motor_gain = 1.0
    motor_freeze_threshold = 65
    pass
if computer_name == 'Mr_Yellow':
    #motor_gain = 0.9
    pass
if computer_name == 'Mr_Black':
    #motor_gain = 1.0
    pass
if computer_name == 'Mr_White':
    #motor_gain = 1.0
    pass

if computer_name == 'Mr_Teal':
    #motor_gain = 1.0
    pass
if computer_name == 'Mr_Audi':
    steer_gain = 1.0
    motor_gain = 1.0
    motor_freeze_threshold = 65
    gyro_freeze_threshold = 1000
    acc_freeze_threshold_x = 20
    acc_freeze_threshold_y = 20
    acc_freeze_threshold_z = 20
    pass
if computer_name == 'Mr_Purple':
    #motor_gain = 1.0
    pass
if computer_name == 'Mr_LightBlue':
    #motor_gain = 1.0
    pass
if computer_name == 'Mr_Blue_Original':
    motor_gain = 0.5
    pass
#
###################################################################
# 

non_user_section = True
if non_user_section:
	if Direct == 1:
		task = 'direct'
	elif Play == 1:
		task = 'play'
	elif Follow == 1:
		task = 'follow'
	elif Furtive == 1:
		task = 'furtive'
	elif Racing == 1:
		task = 'racing'
	else:
		assert(False)
	foldername = ''
	if Follow == 1:
		foldername = 'follow_'
	model_name = weights_file_path.split('/')[-1]
	if AI == 1:
		foldername = foldername + 'AI_' + model_name +'_'
	foldername = foldername + task + '_'
	foldername = foldername + Location + '_'
	foldername = foldername + time_str() + '_'
	foldername = foldername + computer_name

     


