#!/usr/bin/env bash
# 
source /opt/ros/$ROS_DISTRO/setup.bash
# Set default value if no argument is provided
ros2 run rviz2 rviz2 -d ./config/prune_config.rviz