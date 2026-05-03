# OREN: Octree Residual Network for Real-Time Euclidean Signed Distance Mapping



This repository contains the code for the paper: **OREN: Octree Residual Network for Real-Time Euclidean Signed Distance Mapping**.

This repo could run on Replica(NICE-SLAM format) dataset and run mapping as a ROS 2 node.

OREN is a hybrid SDF reconstruction framework that combines gradient-augmented octree interpolation with an implicit neural residual to achieve efficient, continuous non-truncated, and highly accurate Euclidean SDF mapping..



## Installation

### Prerequisites

- Ubuntu (24.04 tested) / Arch Linux
- Python 3.12 (3.10, 3.11 should also work)
- CUDA (tested with CUDA 12.8 and PyTorch 2.8.0)

### Steps

1. Clone the repository
  ```bash
    git clone --recursive https://github.com/ExistentialRobotics/oren.git
    cd oren
  ```
2. Setup pipenv environment
  ```bash
    pip install pipenv  # or sudo apt install pipenv
    pipenv install
    pipenv shell --verbose
  ```
    If you use other virtual environment tools, you can also install the dependencies by
3. Install system dependencies
  - For Ubuntu
    ```bash
    sudo apt install \
        cmake \
        g++ \
        ccache \
        git \
        libeigen3-dev \
        libyaml-cpp-dev \
        libabsl-dev \
        python3-dev \
        python3-pip \
        pybind11-dev
    ```
    - For Arch Linux
    ```bash
    sudo pacman -S --needed \
        cmake \
        gcc \
        ccache \
        git \
        eigen \
        yaml-cpp \
        abseil-cpp \
        python \
        python-pip \
        pybind11
    ```
4. Install other dependencies
  ```bash
    cd deps/pytorch3d
    pip install --no-build-isolation --verbose .
    cd ../..

    cd deps/sparse_octree
    python setup.py install
    cd ../..

    cd deps/erl_geometry
    pip install --no-build-isolation --verbose .
    cd ../..

    # for Arch Linux, when installing pytorch3d and erl_geometry,
    # you may need to set CC, CXX and NVCC_PREPEND_FLAGS to use gcc-14/g++-14
    # CC=gcc-14 CXX=g++-14 NVCC_PREPEND_FLAGS='-ccbin g++-14' pip install --no-build-isolation --verbose .
  ```

## Prepare Dataset

Download Replica Dataset (only mesh, camera parameter and trajectory) at [One-Drive Link](https://ucsdcloud-my.sharepoint.com/my?id=%2Fpersonal%2Fzhdai%5Fucsd%5Fedu%2FDocuments%2FPublicShare%2Fgrad%2Dsdf%2Freplica%2Etar%2Egz&parent=%2Fpersonal%2Fzhdai%5Fucsd%5Fedu%2FDocuments%2FPublicShare%2Fgrad%2Dsdf&ga=1) and put it under path "data/Replica"

Run the following commands to preprocess the Replica dataset:

The script `[oren/oren/dataset/replica_obb_rotation.py](oren/oren/dataset/replica_obb_rotation.py)` is used to rotate mesh and trajectory to better match octree.

```bash
python oren/oren/dataset/replica_obb_rotation.py \
    --dataset-dir data/Replica \
    --output-dir data/Replica_preprocessed
```

copy camera parameter to preprocessed data folder

```bash
cp data/Replica/cam_params.json data/Replica_preprocessed
```

The script `[oren/oren/dataset/replica_augment_views.py](oren/oren/dataset/replica_augment_views.py)` is used to augment the Replica dataset with additional virtual camera views (e.g., upward-looking frames) to improve spatial coverage for training.

```bash
python oren/oren/dataset/replica_augment_views.py \
    --original-dir data/Replica_preprocessed \
    --output-dir data/Replica_preprocessed \
    # --scenes room0  # (optional) Process specific scenes only. If not set, process all scenes. \
    # --interval 50  # (optional, default=50) Insert upward-looking frames every n frames. \
    # --n-rolls-per-insertion 10 # (optional, default=10) Number of roll rotations per insertion. \
    # --keep-existing  # (optional) Keep existing RGBD data.
```

Download our preprocessed Replica dataset

## Run OREN

### Example: Training on Replica

Run the following command to start training on the Replica dataset

```bash
python oren/oren/trainer.py  --config configs/v2/replica/room0.yaml
```

### Run GUI Trainer

The GUI trainer allows interactive visualization and monitoring of the training process, including SDF slice, octree structure, and camera poses.

```bash
python oren/oren/gui_trainer.py \
    --gui-config configs/v2/gui.yaml \
    --trainer-config configs/v2/replica/room0.yaml \
    --gt-mesh-path data/Replica_preprocessed/mesh.ply \
    --apply-offset-to-gt-mesh \
    --copy-scene-bound-to-gui
```

## ROS 2 (this branch)

### Build and install

```bash
# from repo root
colcon build --verbose --symlink-install
source install/setup.bash
```

### Launch mapping node with rosbag

```bash
ros2 launch oren_ros mapping_with_bag.launch.py \
  config_path:=<repo_root>/configs/v2/quad-ros.yaml
```

Optional arguments:

```bash
ros2 launch oren_ros mapping_with_bag.launch.py \
  bag_path:=<newer_college_bag_path> \
  config_path:=<repo_root>/configs/v2/trainer.yaml \
  play_rate:=1.0 \
  bag_delay:=1.0
```

## Docker

### 1. Build the image

First, build the Docker image (make sure you are in the project root):

Use the following command to start a container with GPU, X11 display, and device access enabled:

```bash
./docker/build.bash
```

This script will create the Docker image `erl/oren:24.04`.

### 2. Run the container

Use the following command to start a container with GPU, X11 display, and device access enabled:

```bash
docker run --privileged --restart always -t \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME:$HOME:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    --gpus all \
    --runtime=nvidia \
    -e DISPLAY \
    --net=host \
    --detach \
    --hostname container-oren \
    --add-host=container-oren:127.0.0.1 \
    --name oren \
    erl/oren:24.04 \
    bash -l
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{dai2026oren,
      title={OREN: Octree Residual Network for Real-Time Euclidean Signed Distance Mapping},
      author={Zhirui Dai and Qihao Qian and Tianxing Fan and Nikolay Atanasov},
      year={2026},
      eprint={2510.18999},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.18999},
}
```

## Acknowledgement

- We develop our key frame selection strategy based on [H2-Mapping](https://github.com/Robotics-STAR-Lab/H2-Mapping).
- We create the GUI based on [Open3D](http://www.open3d.org/) with inspirations from [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM).

