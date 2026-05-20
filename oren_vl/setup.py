from setuptools import find_packages, setup

package_name = "oren_vl"

setup(
    name=package_name,
    version="0.2.0",
    author="Zhirui Dai",
    author_email="zhdai@ucsd.edu",
    description="Vision-language extension for OREN.",
    url="https://github.com/ExistentialRobotics/oren",
    license="MIT",
    install_requires=["setuptools"],
    packages=find_packages(),
    zip_safe=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    entry_points={
        "console_scripts": [],
    },
)
