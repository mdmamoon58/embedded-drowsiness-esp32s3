from setuptools import setup, find_packages

setup(
    name="DriverDrowsinessDetection",
    version="0.1.0",
    description="Embedding-based driver drowsiness detection using deep learning using esp32-cam images.",
    author="technoob",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",      
        "matplotlib"
    ],
    python_requires=">=3.8",
)
