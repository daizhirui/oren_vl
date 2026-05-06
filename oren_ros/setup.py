from glob import glob
import os

from setuptools import find_packages, setup

package_name = "oren_ros"


def _collect_configs():
    entries = []
    for dirpath, _, filenames in os.walk("configs", followlinks=True):
        files = [os.path.join(dirpath, f) for f in filenames]
        if files:
            entries.append((os.path.join("share", package_name, dirpath), files))
    return entries



setup(
    name=package_name,
    version="0.1.0",
    author="Zhirui Dai, Qihao Qian",
    author_email="zhdai@ucsd.edu, q2qian@ucsd.edu",
    description="ROS 2 wrapper for the oren algorithm.",
    url="https://github.com/ExistentialRobotics/oren",
    license="MIT",
    install_requires=["setuptools"],
    packages=find_packages(),
    zip_safe=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        *_collect_configs(),
    ],
    entry_points={
        "console_scripts": [
            "mapping_node = oren_ros.node.mapping_node:main",
            "clock_node = oren_ros.node.clock_node:main",
            "sdf_query_node = oren_ros.node.sdf_query_node:main",
        ],
    },
)
