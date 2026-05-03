ARG BASE_IMAGE=erl/grad_sdf:24.04
FROM ${BASE_IMAGE}

ARG TZ=America/Los_Angeles
ARG LOCALE=en_US.UTF-8
ARG APT_MIRROR=""
ARG ROS_APT_SOURCE_VERSION=1.1.0

ENV TZ=${TZ}
ENV LANG=${LOCALE}
ENV LANGUAGE=${LOCALE}
ENV LC_ALL=${LOCALE}

# VIRTUAL_ENV / PATH are inherited from the base image (algorithm layer set
# them up at /opt/venv). PYTHONPATH is added below after ROS is installed so
# the venv site-packages take precedence over apt-installed Python deps,
# while still finding rclpy/sensor_msgs/etc. in /usr/lib/python3/dist-packages.

RUN if [ -n "$APT_MIRROR" ]; then \
        cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.backup && \
        sed -i "s|http://archive.ubuntu.com/ubuntu/|$APT_MIRROR|g" /etc/apt/sources.list.d/ubuntu.sources && \
        sed -i "s|http://security.ubuntu.com/ubuntu/|$APT_MIRROR|g" /etc/apt/sources.list.d/ubuntu.sources; \
    fi

######################
# ROS 2 Jazzy        #
######################

RUN DEBIAN_FRONTEND=noninteractive apt update -y && apt install -y \
        software-properties-common \
        curl \
        gnupg2 \
        lsb-release && \
    add-apt-repository universe -y && \
    curl -L -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb && \
    rm /tmp/ros2-apt-source.deb

RUN DEBIAN_FRONTEND=noninteractive apt update -y && apt upgrade -y && \
    apt install -y \
        ros-jazzy-desktop \
        ros-dev-tools \
        python3-colcon-common-extensions \
        python3-rosdep \
        ros-jazzy-tf2 \
        ros-jazzy-tf2-ros \
        ros-jazzy-tf2-geometry-msgs \
        ros-jazzy-rosbag2-py \
        ros-jazzy-sensor-msgs \
        ros-jazzy-sensor-msgs-py \
        ros-jazzy-nav-msgs \
        ros-jazzy-geometry-msgs \
        ros-jazzy-std-msgs

RUN rosdep init && rosdep update

# Make ROS Python bindings (rclpy, sensor_msgs.msg, ...) visible from the venv
# without copying them. Order matters: venv site-packages first so Torch/numpy
# in the venv shadow any apt-installed copies; then system paths for ROS.
ENV PYTHONPATH=${VIRTUAL_ENV}/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages:/usr/lib/python3/dist-packages

##########################
# SSH and X11 forwarding #
##########################

RUN DEBIAN_FRONTEND=noninteractive apt install -y zsh openssh-server && \
    echo "Port 8022" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "X11Forwarding yes" >> /etc/ssh/sshd_config && \
    echo "X11UseLocalhost no" >> /etc/ssh/sshd_config

ENV GOMP_SPINCOUNT=1
RUN echo "export GOMP_SPINCOUNT=1" >> /etc/environment

# Clean apt lists; restore APT mirror config if it was patched.
RUN apt clean && rm -rf /var/lib/apt/lists/* && \
    if [ -n "$APT_MIRROR" ]; then \
        mv /etc/apt/sources.list.d/ubuntu.sources.backup /etc/apt/sources.list.d/ubuntu.sources; \
    fi

CMD ["bash", "-l"]
