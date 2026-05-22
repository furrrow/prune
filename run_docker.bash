#!/usr/bin/env bash

docker run -it --rm \
  --name prune-deployment \
  --user $(id -u):$(id -g) \
  --net=host \
  --ipc=host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e HOME=/home/ros \
  -e ROS_LOCALHOST_ONLY=0 \
  -e ROS_DISABLE_SHARED_MEMORY=1 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI=file:///etc/cyclonedds-robot.xml \
  -e XDG_CACHE_HOME=/home/ros/.cache \
  -e MPLCONFIGDIR=/home/ros/.cache/matplotlib \
  -v $HOME:/home/ros \
  -v $(pwd):/workspace/prune \
  -v /home/gamma-nav/.config/cyclonedds-robot.xml:/etc/cyclonedds-robot.xml:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v ~/.ssh:/home/ros/.ssh:ro \
  -v ~/.cache/uv:/home/ros/.cache/uv \
  prune-deployment \
  bash
