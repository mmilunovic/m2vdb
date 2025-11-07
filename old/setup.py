# setup.py

from setuptools import setup, find_packages

setup(
    name="m2vdb",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "faiss-cpu",
    ],
    author="mmilunovic",
    description="Lightweight vector DB built for learning purposes",
    python_requires='>=3.7',
)
