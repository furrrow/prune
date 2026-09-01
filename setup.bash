#!/bin/bash

rosmode () {
  if [ "$(cat /sys/class/net/enp132s0/carrier 2>/dev/null)" = "1" ]; then
    export CYCLONEDDS_URI=file://$HOME/.config/cyclonedds-robot.xml
    echo "ROS mode: ROBOT (ethernet)"
  else
    export CYCLONEDDS_URI=file://$HOME/.config/cyclonedds-laptop.xml
    echo "ROS mode: LAPTOP (no ethernet required)"
  fi
  pkill -f ros2daemon 2>/dev/null || true
  rm -rf ~/.ros/daemon
}

alias realsense2="ros2 launch realsense2_camera rs_launch.py"

GHOST_IP=192.168.168.105
# this is gamma's jackal
JACKAL_IP=192.168.131.1
HUSKY_IP=192.168.131.1

export ROBOT_IP=192.168.131.1
export LAPTOP_IP=192.168.131.7


alias ghost="ssh ghost@192.168.168.105"
alias husky="ssh mrc-user@192.168.131.1"


export ROS_DOMAIN_ID=123
# rosmode
source /opt/ros/humble/setup.bash
echo $ROS_DISTRO

# when having trouble with .ssh folder ownership issues...
# chown -R root:root /root/.ssh
# enable offline mode when we are not connected to internet...
export TRANSFORMERS_OFFLINE=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
