import os

from setuptools import find_packages, setup

package_name = "oren"


def _collect_configs():
    entries = []
    for dirpath, _, filenames in os.walk("configs", followlinks=True):
        files = [os.path.join(dirpath, f) for f in filenames]
        if files:
            entries.append((os.path.join("share", package_name, dirpath), files))
    return entries


setup(
    name=package_name,
    version="0.2.0",
    author="Zhirui Dai, Qihao Qian",
    author_email="zhdai@ucsd.edu, q2qian@ucsd.edu",
    description="Gradient-augmented octree + neural residual SDF reconstruction (algorithm package)",
    url="https://github.com/ExistentialRobotics/oren",
    install_requires=["setuptools"],
    license="MIT",
    packages=find_packages(),
    zip_safe=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        *_collect_configs(),
    ],
)
