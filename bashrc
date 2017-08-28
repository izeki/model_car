############# data collection shortcuts (smartphone login) #######
#
# cd; cd kzpy3; git pull; cd
alias ls='ls -al --color=auto'
alias rhz='rostopic hz /bair_car/zed/left/image_rect_color'
alias rls='ls -al /media/ubuntu/rosbags'
alias rrm='rm catkin_ws/src/bair_car/rosbags/*'
alias rlog='rm -r ~/.ros/log/*'
alias rla='roslaunch bair_car bair_car.launch use_zed:=true record:=true'
alias rlai='roslaunch bair_car bair_car.launch use_zed:=true record:=true use_AI:=true'
alias rli='roslaunch bair_car bair_car.launch use_zed:=true record:=false use_AI:=true'
alias rlab='(rla 3>&1 1>&2 2>&3 | grep -v "slMat2cvMat: mat type currently not supported 0 0") 3>&1 1>&2 2>&3'
#alias rlac='roslaunch bair_car bair_car.launch use_zed:=true record:=true caffe:=true'
#alias rgla='cd ~/kzpy3;git pull;cd;roslaunch bair_car bair_car.launch use_zed:=true record:=true'
alias rgla='cd ~/model_car;git pull;cd;roslaunch bair_car bair_car.launch use_zed:=true record:=true'
alias rglna='cd ~/model_car;git pull;cd;roslaunch bair_car bair_car.launch use_zed:=true record:=false'
alias rglai='cd ~/model_car;git pull;cd;roslaunch bair_car bair_car.launch use_zed:=true record:=true use_AI:=true'
alias rglnai='cd ~/model_car;git pull;cd;roslaunch bair_car bair_car.launch use_zed:=true record:=false use_AI:=true'
alias rgy='rostopic echo /bair_car/gyro'
alias rgp='rostopic echo /bair_car/gps'
alias rac='rostopic echo /bair_car/acc'
alias rst='rostopic echo /bair_car/steer'
alias rmo='rostopic echo /bair_car/motor'
alias ren='rostopic echo /bair_car/encoder'
alias rcd='cd /media/ubuntu/rosbags'
alias ssd='sudo shutdown -h now'
alias killros='killall python && killall roslaunch && killall record'
alias rcn='echo $COMPUTER_NAME'
alias rivl='rosrun image_view image_view image:=/bair_car/zed/left/image_rect_color'
alias rivr='rosrun image_view image_view image:=/bair_car/zed/right/image_rect_color'
alias rlat='rostopic echo /bair_car/GPS2_lat'
alias rstat='python ~/model_car/rosstatus.py'
alias test_caffe='cd ~/caffe; build/tools/caffe time --model=models/bvlc_alexnet/deploy.prototxt --gpu=0'
alias zed_explorer='/usr/local/zed/tools/ZED\ Explorer'
# sudo vi /etc/hosts # to edit hosts
alias speedup='echo bdd_path123 | sudo -S ~/jetson_clocks.sh'
alias setswap='echo bdd_path123 | sudo -S ~/git/postFlashTX1/createSwapfile.sh -d /media/ubuntu/rosbags/ -s 8'
alias mrsil='ssh ubuntu@Mr_Silver'
alias mryel='ssh ubuntu@Mr_Yellow'
alias mrblu='ssh ubuntu@Mr_Blue'
alias mrora='ssh ubuntu@Mr_Orange'
alias mrtea='ssh ubuntu@Mr_Teal'
alias mrbla='ssh ubuntu@Mr_Black'
alias mrwhi='ssh ubuntu@Mr_White'
alias ssh_date='sudo date --set="$(ssh karlzipser@192.168.1.16 date)"'
alias fixScreen='DISPLAY=:0 xrandr --output HDMI-0 --mode 1024x768'


######################## for .bashrc from MacBook #################
#
##echo "source ~/8August2016_common_aliases"
##export DISPLAY=:0.0
#alias gacp="git add .;git commit -m 'gacp';git push origin master"
#alias gckzpy3="git clone https://github.com/karlzipser/kzpy3.0.git"
#git config --global credential.helper "cache --timeout=86400"
#alias ipy="ipython --no-banner"
#export PYTHONPATH=~:$PYTHONPATH
##export PYTHONPATH=~/kzpy3/caf/layers:$PYTHONPATH
##export PYTHONPATH=~/kzpy3/caf2/layers:$PYTHONPATH
#export PYTHONSTARTUP=~/kzpy3/vis.py
#export PATH=~/kzpy3/scripts:$PATH
#export PATH=~/caffe/build/tools:$PATH

#export PYTHONPATH=~/caffe/python:$PYTHONPATH

# ARUCO CODE PYTHON PATH CHANGE

##if [ "$(whoami)" == "ubuntu" ]
##then
#export PYTHONPATH=$PYTHONPATH:~/kzpy3/data_analysis
#export PYTHONPATH=$PYTHONPATH:~/kzpy3/data_analysis/trajectory_generator
#export PYTHONPATH=$PYTHONPATH:~/kzpy3/data_analysis/visualization
##fi
# ARUCO CODE PYTHON PATH CHANGE

export PYTHONPATH="$HOME/model_car:$PYTHONPATH"
