#!/usr/bin/env bash
# 
# docker build -t prune:dev .
docker run -it --rm \
  --name prune-dev \
  --net=host \
  --ipc=host \
  --gpus=all \
  -e ROS_LOCALHOST_ONLY=0 \
  -e ROS_DISABLE_SHARED_MEMORY=1 \
  -e DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ./:/workspace/prune \
  -v ~/.ssh:/root/.ssh\
  prune:dev