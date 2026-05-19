FROM osrf/ros:humble-desktop

# 1. Install basic tools
RUN apt-get update && apt-get install -y \
    curl ca-certificates wget git build-essential python3-numpy vim openssh-client vim tmux

# 2. Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# 3. Set working directory
WORKDIR /workspace

# 4. Entry point: source ROS
RUN sed -i '/^exec "\$@"/i \
if [ -f /workspace/prune/setup.bash ]; then\n\
    source /workspace/prune/setup.bash\n\
fi\n' /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]