from setuptools import setup, find_packages
from glob import glob


setup(
    name="torchdaiv",
    version="0.0.1",
    description="Torch Wrapper for dAiv's lecture",
    author="daiv",
    author_email="cnudaiv@gmail.com",
    url="https://github.com/dAiv-CNU/torchdaiv",
    packages=find_packages("src/main"),
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "rich"
    ]
)
