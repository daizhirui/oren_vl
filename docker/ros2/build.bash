#!/usr/bin/env bash

set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)
REPO_ROOT=$(cd ${SCRIPT_DIR}/../..; pwd)

BASE_IMAGE="${BASE_IMAGE:-erl/oren:24.04}"
ROS_TAG="${ROS_TAG:-erl/oren_ros:jazzy}"

ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
ROS_APT_SOURCE_VERSION=${ROS_APT_SOURCE_VERSION:-1.1.0}

echo "[build] ${ROS_TAG} (base: ${BASE_IMAGE})"
docker build --rm -t ${ROS_TAG} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg ROS_APT_SOURCE_VERSION=${ROS_APT_SOURCE_VERSION} \
    --build-arg APT_MIRROR=${APT_MIRROR} \
    -f ${SCRIPT_DIR}/Dockerfile \
    ${REPO_ROOT} "$@"
