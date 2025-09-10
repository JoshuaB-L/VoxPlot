#!/usr/bin/env python3
"""
Setup script for VoxPlot: Advanced Voxel-based Forest Structure Analysis

This setup script allows VoxPlot to be installed as a Python package,
making it easier to use and distribute.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "PyYAML>=6.0"
    ]

# Development requirements
dev_requirements = [
    "pytest>=6.2.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
    "mypy>=0.931",
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0"
]

setup(
    name="voxplot",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Advanced voxel-based forest structure analysis and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/voxplot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "performance": [
            "numba>=0.56.0",
            "dask>=2022.1.0"
        ],
        "data": [
            "h5py>=3.6.0",
            "netCDF4>=1.5.8",
            "xarray>=0.20.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "voxplot=main:main",
            "voxplot-config=config_manager:create_example_config",
        ],
    },
    include_package_data=True,
    package_data={
        "voxplot": [
            "examples/*.yaml",
            "examples/*.csv",
            "docs/*.md"
        ]
    },
    keywords=[
        "forest structure",
        "lidar",
        "voxel analysis", 
        "leaf area density",
        "wood area density",
        "plant area density",
        "crown analysis",
        "forest modeling",
        "remote sensing",
        "forestry",
        "ecology"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-repo/voxplot/issues",
        "Source": "https://github.com/your-repo/voxplot",
        "Documentation": "https://voxplot.readthedocs.io/",
        "Funding": "https://your-funding-source.org"
    },
    zip_safe=False,
)