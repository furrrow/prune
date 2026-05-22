FROM osrf/ros:humble-desktop

# 1. Install basic tools + CycloneDDS RMW
RUN apt-get update && apt-get install -y \
    curl ca-certificates wget git build-essential \
    python3-numpy vim openssh-client tmux \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# 3. Set CycloneDDS as default RMW
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=file:///dev/null

# 4. Working directory
WORKDIR /workspace

# 5. Entry point: source ROS + workspace
RUN sed -i '/^exec "\$@"/i \
if [ -f /workspace/prune/setup.bash ]; then\n\
    source /workspace/prune/setup.bash\n\
fi\n' /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]