from setuptools import setup
from glob import glob


setup(
    name="torchdaiv",
    version="0.0.1",
    description="Torch Wrapper for dAiv's lecture",
    author="daiv",
    author_email="cnudaiv@gmail.com",
    url="https://github.com/dAiv-CNU/torchdaiv",
    packages=list(set(['/'.join(path.removesuffix(".py").replace("\\", "/").split('/')[:-1]) for path in glob("src/main/**/*.py", recursive=True)])),
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "rich"
    ]
)
