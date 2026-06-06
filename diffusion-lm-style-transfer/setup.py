from setuptools import setup, find_packages

setup(
    name="diffusion-lm-style-transfer",
    version="0.1.0",
    packages=find_packages(include=["phase*"]),
    python_requires=">=3.10",
)
