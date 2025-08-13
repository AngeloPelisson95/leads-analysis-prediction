from setuptools import find_packages, setup

setup(
    name="lead-generation-vehicle-listings",
    packages=find_packages(),
    version="0.1.0",
    description="Lead Generation Analysis and Optimization in Vehicle Listings",
    author="Angelo Pelisson",
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
    ],
)
