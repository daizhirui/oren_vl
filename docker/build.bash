#!/usr/bin/env bash

set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)
REPO_ROOT=$(cd ${SCRIPT_DIR}/..; pwd)

ALGO_TAG="erl/grad_sdf:24.04"
ROS_TAG="erl/grad_sdf_ros:jazzy"

ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
ROS_APT_SOURCE_VERSION=${ROS_APT_SOURCE_VERSION:-1.1.0}

# Layer 1 — algorithm image (CUDA + Python deps + grad_sdf)
echo "[build] Layer 1: ${ALGO_TAG}"
if [ -z "${APT_MIRROR}" ]; then
    docker build --rm -t ${ALGO_TAG} -f ${SCRIPT_DIR}/grad_sdf/Dockerfile ${REPO_ROOT} "$@"
else
    docker build --rm -t ${ALGO_TAG} --build-arg APT_MIRROR=${APT_MIRROR} -f ${SCRIPT_DIR}/grad_sdf/Dockerfile ${REPO_ROOT} "$@"
fi

# Layer 2 — ROS 2 Jazzy + colcon build of grad_sdf / grad_sdf_msgs / grad_sdf_ros
echo "[build] Layer 2: ${ROS_TAG}"
docker build --rm -t ${ROS_TAG} \
    --build-arg BASE_IMAGE=${ALGO_TAG} \
    --build-arg ROS_APT_SOURCE_VERSION=${ROS_APT_SOURCE_VERSION} \
    --build-arg APT_MIRROR=${APT_MIRROR} \
    -f ${SCRIPT_DIR}/ros2/Dockerfile \
    ${REPO_ROOT} "$@"
