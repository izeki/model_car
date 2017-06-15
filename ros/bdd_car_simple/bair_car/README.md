#################
### Setup ROS ###
#################

Add to .bashrc: source /opt/ros/indigo/setup.bash

$ mkdir -p ~/catkin_ws/src

$ cd ~/catkin_ws/src

$ catkin_init_workspace

$ cd ~/catkin_ws/

$ catkin_make

Add to .bashrc: source ~/catkin_ws/devel/setup.bash

#####################
### Setup ZED ROS ###
#####################

$ sudo apt-get install libpcl-1.7-all ros-indigo-pcl-ros ros-indigo-image-view

$ git clone https://github.com/stereolabs/zed-ros-wrapper.git

$ cd ~/catkin_ws

$ catkin_make

Test that it works

$ roslaunch zed_wrapper zed.launch

$ rosrun image_view image_view image:=/camera/right/image_rect_color

######################
### Setup our code ###
######################

$ git clone https://github.com/izeki/model_car.git

$ roslaunch bair_car bair_car.launch use_zed:=true

$ rosrun image_view image_view image:=/bair_car/zed/right/image_rect_color

And rostopic echo all the topics to check

###########################
### Running experiments ###
###########################

Go into bair_car.launch and change the bagpath to where you want to record data to (e.g. flash drive)

$ roslaunch bair_car bair_car.launch use_zed:=true record:=true
