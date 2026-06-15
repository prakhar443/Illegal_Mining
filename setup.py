from setuptools import find_packages, setup

setup(
    name="spearnet",
    version="0.1.0",
    description="SPEAR-Net: lightweight color-prior-guided recall-optimized mining-structure segmentation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
