from glob import glob

from setuptools import find_packages, setup

package_name = "oren_ros"

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
    ],
    entry_points={
        "console_scripts": [
            "mapping_node = oren_ros.node.mapping_node:main",
        ],
    },
)
