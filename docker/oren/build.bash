#!/usr/bin/env bash

set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)
REPO_ROOT=$(cd ${SCRIPT_DIR}/../..; pwd)

ALGO_TAG="${ALGO_TAG:-erl/oren:24.04}"

echo "[build] ${ALGO_TAG}"
if [ -z "${APT_MIRROR}" ]; then
    docker build --rm -t ${ALGO_TAG} -f ${SCRIPT_DIR}/Dockerfile ${REPO_ROOT} "$@"
else
    docker build --rm -t ${ALGO_TAG} --build-arg APT_MIRROR=${APT_MIRROR} -f ${SCRIPT_DIR}/Dockerfile ${REPO_ROOT} "$@"
fi
