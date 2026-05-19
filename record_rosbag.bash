#!/bin/bash

# bag_name
BAG_NAME=${1:-rosbag_record_name}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -z "$1" ]; then
	read -p "Enter bag name to be recorded: " BAG_NAME
	if [ -z "$BAG_NAME" ]; then
		echo "Bag name required, exiting..."
		exit 1
	fi
else
	BAG_NAME=$1
fi

OUTPUT_NAME="${BAG_NAME}_${TIMESTAMP}"
echo "Recording rosbag: $OUTPUT_NAME"

ros2 bag record \
	-o "$OUTPUT_NAME" \
	/odom_lidar \
	/os_cloud_node/metadata \
	/os_cloud_node/os_driver/transition_event \
	/os_cloud_node/points_shm \
	/os_cloud_node/points \
	/mcu/state/imu \
	/mcu/command/manual_twist \
	/camera/camera/color/image_raw \
	/camera/camera/color/camera_info \
	/tf \
	/tf_static \




