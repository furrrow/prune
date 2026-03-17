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

echo "sourcing ros humble"
source /opt/ros/humble/setup.bash

alias realsense2="ros2 launch realsense2_camera rs_launch.py"
alias chop='docker run -it --rm \
  --name chop-dev \
  --net=host \
  --ipc=host \
  --gpus=all \
  -e ROS_LOCALHOST_ONLY=0 \
  -e ROS_DISABLE_SHARED_MEMORY=1 \
  -v /home/gamma-nav/Documents/Projects/git_repos/CHOP:/workspace/chop \
  chop:dev'

export ROBOT_IP=192.168.131.1
export LAPTOP_IP=192.168.131.7

alias bridge_husky="docker run -it --rm \
  --name bridge-husky \
  --net=host \
  --ipc=host \
  --gpus=all \
  -e ROS_MASTER_URI=http://$ROBOT_IP:11311 \
  -e ROS_IP=$LAPTOP_IP \
  -e ROS_DOMAIN_ID=0 \
  -e ROBOT_IP=192.168.131.1 \
  -e LAPTOP_IP=192.168.131.7 \
  -e ROS_LOCALHOST_ONLY=0 \
  -e ROS_DISABLE_SHARED_MEMORY=1 \
  --device=/dev/ttyACM0:/dev/ttyACM0 \
  ros1_ros2_bridge:noetic_humble
"

alias bridge_husky_exec='docker exec -it bridge-husky /bin/bash'

#alias chop='docker run -it --rm \
#  --name chop-dev \
#  --net=host \
#  --ipc=host \
#  --gpus=all \
#  -e ROS_LOCALHOST_ONLY=0 \
#  -e ROS_DISABLE_SHARED_MEMORY=1 \
#  -e CYCLONEDDS_URI=file:///etc/cyclonedds.xml \
#  -v /home/gamma-nav/Documents/Projects/git_repos/CHOP:/workspace/chop \
#  -v /home/gamma-nav/cyclonedds.xml:/etc/cyclonedds.xml:ro \
#  chop:dev'

alias chop_exec='docker exec -it chop-dev /bin/bash'
alias ghost="ssh ghost@192.168.168.105"
alias husky="ssh mrc-user@192.168.131.1"
alias deac="conda deactivate"
GHOST_IP=192.168.168.105
# this is gamma's jackal
JACKAL_IP=192.168.131.1
HUSKY_IP=192.168.131.1

export ROS_DOMAIN_ID=123
rosmode
# ROS_DOMAIN_ID:
# ghost: 123
# jackal: 0
# husky: ??
