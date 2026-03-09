# Start with the official ROS Noetic base image
FROM osrf/ros:noetic-desktop

# Install pip and basic system tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch, Diffusers, and the AI libraries you need
# (We install the standard stable versions compatible with Python 3.8)
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    diffusers \
    numpy \
    pillow \
    pyyaml \
    scipy

# Install AI Dependencies
RUN pip3 install efficientnet_pytorch \
    vit_pytorch==1.11.0 \
    wandb==0.15.0 \
    prettytable \
    tqdm \
    rosbags

# Set the default folder when you enter the container
WORKDIR /workspace

# 3. Copy the requirements list we made
#COPY requirements.txt /workspace/requirements.txt

# 4. Install everything in one go
#RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Source ROS automatically every time you open a terminal in the container
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc