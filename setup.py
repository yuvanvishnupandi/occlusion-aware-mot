from setuptools import setup, find_packages

setup(
    name="occlusion-aware-mot",
    packages=find_packages(),
    version="1.0.0",
    license="MIT",
    description="Identity-Preserving Multi-Object Tracking through Dense Occlusion",
    author="Yuvan Vishnu Pandi",
    url="https://github.com/yuvanvishnupandi/occlusion-aware-mot",
    keywords=[
        "computer vision",
        "multi-object tracking",
        "deep learning",
        "kalman filter",
        "re-identification",
        "dense occlusion"
    ],
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "opencv-python>=4.8.0",
        "onnxruntime>=1.15.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
