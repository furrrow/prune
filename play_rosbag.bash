#!/usr/bin/env bash
# 
source /opt/ros/$ROS_DISTRO/setup.bash
# Set default value if no argument is provided
bag_name="${1:-iribe_5207}"
echo "playing bag: $bag_name"
ros2 bag play ./deployment/topomaps/bags/$bag_name/*.db3