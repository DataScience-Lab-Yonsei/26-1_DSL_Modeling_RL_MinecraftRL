"""
Setup script for cuboid_house_rl.

Usage:
    pip install -e .           # Development/editable install (recommended)
    pip install .              # Standard install
"""
from setuptools import setup, find_packages

setup(
    name="cuboid_house_rl",
    version="0.3.0",
    description="RL agent for building cuboid houses in Minecraft via CraftGround",
    packages=find_packages(include=["cuboid_house_rl", "cuboid_house_rl.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "gymnasium>=0.29",
        "wandb>=0.16",
        "craftground>=2.6.15",
    ],
)
