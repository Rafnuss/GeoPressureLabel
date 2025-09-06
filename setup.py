#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="GeoPressureLabel",
    version="0.0.1",
    description="Downloads measurement label files from Zenodo, extracts sensor data, and builds a neural network to predict label values.",
    author="Raphaël Nussbaumer",
    author_email="rafnuss@gmail.com",
    url="https://github.com/Rafnuss/GeoPressureLabel",
    install_requires=[
        "lightning",
        "hydra-core",
        "requests",
        "pandas",
        "numpy",
        "tqdm",
        "scikit-learn",
        "torch",
        "matplotlib",
        "seaborn",
    ],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "download_command = src.download_data:main",
            "process_command = src.process_data:main",
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
