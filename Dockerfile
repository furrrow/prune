FROM osrf/ros:humble-desktop

# --------------------------
# 1. Install basic tools
# --------------------------
RUN apt-get update && apt-get install -y \
    wget git build-essential python3-numpy vim openssh-client tmux

# --------------------------
# 2. Install Miniconda
# --------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV CONDA_DIR=/opt/miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Accept TOS and Update conda base
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -n base -c defaults -y conda


# --------------------------
# 3. Create project environment
# --------------------------
RUN conda create -y -n prune python=3.10

# --------------------------
# 4. Configure Conda environment
# --------------------------
ENV CONDA_DEFAULT_ENV=prune
ENV PATH=$CONDA_DIR/envs/$CONDA_DEFAULT_ENV/bin:$PATH
ENV PYTHONPATH=$CONDA_DIR/envs/$CONDA_DEFAULT_ENV/lib/python3.10/site-packages:$PYTHONPATH

# --------------------------
# 5. Set working directory
# --------------------------
WORKDIR /workspace

# --------------------------
# 6. Install project in editable mode
# --------------------------
# Assume your project source code is in /workspace/prune
# Use conda run to ensure pip runs inside the conda env
#RUN conda run -n prune pip install --upgrade pip && \
#    conda run -n prune pip install -e /workspace/prune

# --------------------------
# 7. Entry point: source ROS
# --------------------------
#ENTRYPOINT ["/bin/bash", "-c", "source /ros_entrypoint.sh && source /workspace/prune/.devcontainer/setup.bash && exec bash"]
RUN echo 'source /workspace/prune/.devcontainer/setup.bash'       >> /ros_entrypoint.sh && \
    echo 'source "'"$CONDA_DIR"'/bin/activate" "'"$CONDA_DEFAULT_ENV"'"' >> /ros_entrypoint.sh && \
    echo 'export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH' >> /ros_entrypoint.sh && \
    echo 'exec "$@"'                               >> /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]