from setuptools import setup, find_packages

setup(
    name="workout_foundation",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "fitparse>=1.2.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "scipy>=1.10.0",
        "dash>=2.14.0",
        "dash-bootstrap-components>=1.4.0",
        "pyarrow>=10.0.0",
        "openpyxl>=3.1.0",
    ],
    python_requires=">=3.8",
)
