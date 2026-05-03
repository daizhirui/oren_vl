FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

ARG TZ=America/Los_Angeles
ARG LOCALE=en_US.UTF-8
ARG APT_MIRROR=""

ENV TZ=${TZ}
ENV LANG=${LOCALE}
ENV LANGUAGE=${LOCALE}
ENV LC_ALL=${LOCALE}

RUN cp /etc/apt/sources.list /etc/apt/sources.list.backup &&\
    if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu/|$APT_MIRROR|g" /etc/apt/sources.list; \
    fi
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    DEBIAN_FRONTEND=noninteractive apt update && apt install -y locales && locale-gen en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    sudo \
    apt-utils \
    software-properties-common \
    lsb-release \
    ca-certificates \
    wget \
    curl \
    mesa-utils \
    libgl1 \
    libglx-mesa0 \
    x11-apps \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    cmake \
    g++ \
    ccache \
    git \
    libeigen3-dev \
    libyaml-cpp-dev \
    libabsl-dev \
    pybind11-dev \
    zsh

ENV CMAKE_CUDA_ARCHITECTURES="86;89;90;120"
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0"
# for pytorch3d
ENV FORCE_CUDA=1

# Python venv at /opt/venv, overlaid via PATH so all subsequent RUN/CMD/console-scripts
# resolve `python3` / `pip` to the venv without needing to source activate.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV} && ${VIRTUAL_ENV}/bin/pip install -U pip
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Pre-install everything that does NOT depend on the grad_sdf source: third-party
# Python deps, pytorch3d (from a git URL), and the two C++/CUDA submodules in deps/.
# Copy only the files needed for these installs — not the algorithm source.
COPY requirements.txt /opt/build/requirements.txt
RUN . ${VIRTUAL_ENV}/bin/activate && pip install -r /opt/build/requirements.txt

COPY deps/pytorch3d /opt/build/deps/pytorch3d
RUN cd /opt/build/deps/pytorch3d && \
    rm -rf build || true && \
    . ${VIRTUAL_ENV}/bin/activate && \
    pip install --no-build-isolation --verbose .

COPY deps/sparse_octree /opt/build/deps/sparse_octree
RUN cd /opt/build/deps/sparse_octree && \
    rm -rf build || true && \
    . ${VIRTUAL_ENV}/bin/activate && \
    python3 setup.py install --verbose

COPY deps/erl_geometry /opt/build/deps/erl_geometry
RUN cd /opt/build/deps/erl_geometry && \
    rm -rf build || true && \
    . ${VIRTUAL_ENV}/bin/activate && \
    pip install --no-build-isolation --verbose .

RUN rm -rf /opt/build
RUN userdel -r -f ubuntu || true

# clean
RUN apt clean && rm -rf /var/lib/apt/lists/* && \
    if [ -n "$APT_MIRROR" ]; then \
        mv /etc/apt/sources.list.backup /etc/apt/sources.list; \
    fi

CMD ["bash", "-l"]
