#!/usr/bin/env python
"""
reed to run roslaunch first, e.g.,

roslaunch bair_car bair_car.launch use_zed:=true record:=false
"""
import sys
import traceback
import runtime_parameters as rp
from keras_code import *
try:    
    ########################################################
    #          ROSPY SETUP SECTION
    import roslib
    import std_msgs.msg
    import geometry_msgs.msg
    import cv2
    from cv_bridge import CvBridge,CvBridgeError
    import rospy
    from sensor_msgs.msg import Image
    bridge = CvBridge()
    rospy.init_node('listener',anonymous=True)

    left_list = []
    right_list = []
   
    state = 0
    previous_state = 0
    state_transition_time_s = 0

    
    def state_callback(data):
        global state, previous_state
        if state != data.data:
            if state in [3,5,6,7] and previous_state in [3,5,6,7]:
                pass
            else:
                previous_state = state
        state = data.data

        
    def right_image_callback(data):
        global left_list, right_list, solver
        cimg = bridge.imgmsg_to_cv2(data,"bgr8")
        if len(right_list) > nframes + 3:
            right_list = right_list[-(nframes + 3):]
        right_list.append(cimg)        

        
    def left_image_callback(data):
        global left_list, right_list
        cimg = bridge.imgmsg_to_cv2(data,"bgr8")
        if len(left_list) > nframes + 3:
            left_list = left_list[-(nframes + 3):]
        left_list.append(cimg)
        
        
    def state_transition_time_s_callback(data):
        global state_transition_time_s
        state_transition_time_s = data.data


    GPS2_lat = -999.99
    GPS2_long = -999.99
    GPS2_lat_orig = -999.99
    GPS2_long_orig = -999.99
    def GPS2_lat_callback(msg):
        global GPS2_lat
        GPS2_lat = msg.data
        
    def GPS2_long_callback(msg):
        global GPS2_long
        GPS2_long = msg.data

    camera_heading = 49.0

    
    def camera_heading_callback(msg):
        global camera_heading
        c = msg.data
        #print camera_heading
        if c > 90:
            c = 90
        if c < -90:
            c = -90
        c += 90
        c /= 180.
        
        c *= 99

        if c < 0:
            c = 0
        if c > 99:
            c = 99
        c = 99-c
        camera_heading = int(c)

    freeze = False
    def gyro_callback(msg):
        global freeze
        gyro = msg
        #if np.abs(gyro.y) > gyro_freeze_threshold:
        #    freeze = True
        if np.sqrt(gyro.y**2+gyro.z**2) > rp.gyro_freeze_threshold:
            freeze = True
           
        
    def acc_callback(msg):
        global freeze
        acc = msg
        if np.abs(acc.z) > rp.acc_freeze_threshold_z:
            freeze = True
        if acc.y < rp.acc_freeze_threshold_z_neg:
            freeze = True
        if np.abs(acc.x) > rp.acc_freeze_threshold_x:
            freeze = True
        #if np.abs(acc.y) > acc_freeze_threshold_y:
        #    freeze = True

        
    encoder_list = []
    def encoder_callback(msg):
        global encoder_list
        encoder_list.append(msg.data)
        if len(encoder_list) > 30:
            encoder_list = encoder_list[-30:]

    ##
    ########################################################

    import thread
    import time


    rospy.Subscriber("/bair_car/zed/right/image_rect_color",Image,right_image_callback,queue_size = 1)
    rospy.Subscriber("/bair_car/zed/left/image_rect_color",Image,left_image_callback,queue_size = 1)
    rospy.Subscriber('/bair_car/state', std_msgs.msg.Int32,state_callback)
    rospy.Subscriber('/bair_car/state_transition_time_s', std_msgs.msg.Int32, state_transition_time_s_callback)
    steer_cmd_pub = rospy.Publisher('cmd/steer', std_msgs.msg.Int32, queue_size=100)
    motor_cmd_pub = rospy.Publisher('cmd/motor', std_msgs.msg.Int32, queue_size=100)
    freeze_cmd_pub = rospy.Publisher('cmd/freeze', std_msgs.msg.Int32, queue_size=100)
    model_name_pub = rospy.Publisher('/bair_car/model_name', std_msgs.msg.String, queue_size=10)
    #rospy.Subscriber('/bair_car/GPS2_lat', std_msgs.msg.Float32, callback=GPS2_lat_callback)
    #rospy.Subscriber('/bair_car/GPS2_long', std_msgs.msg.Float32, callback=GPS2_long_callback)
    #rospy.Subscriber('/bair_car/GPS2_lat_orig', std_msgs.msg.Float32, callback=GPS2_lat_callback)
    #rospy.Subscriber('/bair_car/GPS2_long_orig', std_msgs.msg.Float32, callback=GPS2_long_callback)
    #rospy.Subscriber('/bair_car/camera_heading', std_msgs.msg.Float32, callback=camera_heading_callback)
    rospy.Subscriber('/bair_car/gyro', geometry_msgs.msg.Vector3, callback=gyro_callback)
    rospy.Subscriber('/bair_car/acc', geometry_msgs.msg.Vector3, callback=acc_callback)
    rospy.Subscriber('encoder', std_msgs.msg.Float32, callback=encoder_callback)

    ctr = 0


    #from kzpy3.teg2.global_run_params import *

    t0 = time.time()
    time_step = Timer(1)
    AI_enter_timer = Timer(2)
    folder_display_timer = Timer(30)
    git_pull_timer = Timer(60)
    reload_timer = Timer(10)
    AI_steer_previous = 49
    AI_motor_previous = 49
    #verbose = False
    
    
    while not rospy.is_shutdown():
        #state = 3 
        if state in [3,5,6,7]:
            
            if (previous_state not in [3,5,6,7]):
                previous_state = state
                AI_enter_timer.reset()
            if rp.use_AI:
                if not AI_enter_timer.check():
                    #print AI_enter_timer.check()
                    print "waiting before entering AI mode..."
                    steer_cmd_pub.publish(std_msgs.msg.Int32(49))
                    motor_cmd_pub.publish(std_msgs.msg.Int32(49))
                    time.sleep(0.1)
                    continue
                else:
                    if len(left_list) > nframes + 2:
                        
                        camera_data = format_camera_data(left_list, right_list)
                        metadata = format_metadata({'Racing':rp.Racing, 
                                                    'AI':0, 
                                                    'Follow':rp.Follow,
                                                    'Direct':rp.Direct, 
                                                    'Play':rp.Play, 
                                                    'Furtive':rp.Furtive})
                        [AI_steer, AI_motor] = run_model(camera_data, metadata)                        
                        

                        if AI_motor > rp.motor_freeze_threshold and np.array(encoder_list[0:3]).mean() > 1 and np.array(encoder_list[-3:]).mean()<0.2 and state_transition_time_s > 1:
                            freeze = True

                        
                        if freeze:
                            print "######### FREEZE ###########"
                            AI_steer = 49
                            AI_motor = 49

                        freeze_cmd_pub.publish(std_msgs.msg.Int32(freeze))

                         
                        if state in [3,6]:            
                            steer_cmd_pub.publish(std_msgs.msg.Int32(AI_steer))
                        
                        if state in [6,7]:
                            motor_cmd_pub.publish(std_msgs.msg.Int32(AI_motor))

                        
                        if True: #verbose:
                            print("{},{},{},{}".format(AI_motor,AI_steer,rp.motor_gain,rp.steer_gain,state))
                            #print AI_motor,AI_steer,motor_gain,steer_gain,state

        else:
            AI_enter_timer.reset()
            if state == 4:
                freeze = False
            if state == 2:
                freeze = False
            if state == 1:
                freeze = False
            if state == 4 and state_transition_time_s > 30:
                print("Shutting down because in state 4 for 30+ s")
                #unix('sudo shutdown -h now')
        if time_step.check():
            print(d2s("In state",state,"for",state_transition_time_s,"seconds, previous_state =",previous_state))
            time_step.reset()
            if not folder_display_timer.check():
                print("*** Data foldername = "+rp.foldername+ '***')
        if reload_timer.check():            
            reload(rp)
            
            model_name_pub.publish(std_msgs.msg.String(rp.weights_file_path))
            reload_timer.reset()

        if git_pull_timer.check():
            #unix(opjh('kzpy3/kzpy3_git_pull.sh'))
            unix(opjh('model_car/model_car_git_pull.sh'))
            git_pull_timer.reset()

except Exception as e:
    print("********** Exception ***********************",'red')
    traceback.print_exc(file=sys.stdout)
    rospy.signal_shutdown(d2s(e.message,e.args))

