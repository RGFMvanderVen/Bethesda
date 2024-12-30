from setuptools import setup, find_packages

base_packages = [
    "polyfuzz>=0.3.4",
    "nltk>=3.6",
    "pandas>=1.3.0",
    "tqdm>=4.26.0",
    "scikit-learn>=1.0.1",
    "openpyxl>=3.0.7",
]

test_packages = [
    "pytest>=5.2",
    "pytest-cov>=2.6.1",
    "black>=22.1.0",
    "flake8>=4.0.1",
    "pre-commit>=2.17.0",
]

dev_packages = ["jupyterlab>=3.1.10"] + base_packages + test_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="matcher",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version="0.1.0",
    author="Maarten P. Grootendorst",
    description="Extract Bethesda score from PALGA data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=base_packages,
    extras_require={"test": test_packages, "dev": dev_packages},
    python_requires=">=3.7",
)
