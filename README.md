# Diffusion entropy Analysis

[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square&labelColor=393f46)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=flat-square&labelColor=393f46)](https://github.com/astral-sh/uv)

Diffusion Entropy Analysis is a time-series analysis method for detecting temporal scaling in a data set, such as particle motion, a seismograph, or an electroencephalograph signal. Diffusion Entropy Analysis converts a timeseries into a diffusion trajectory and uses the entropy of this trajectory to measure the temporal scaling in the data. This is accomplished by moving a window along the trajectory, then using the relationship between the natural logarithm of the length of the window and the Shannon entropy to extract the scaling of the time-series process.

For further details about the method and how it works, please see []()

## Installation and use

You'll need an up to date Python installation, e.g., through [uv](https://docs.astral.sh/uv/#python-management).

## Build with

[![numpy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scipy](https://img.shields.io/badge/scipy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![polars](https://img.shields.io/badge/polars-CD792C?style=for-the-badge&logo=polars&logoColor=white)](https://pola.rs/)
[![matplotlib](https://img.shields.io/badge/matplotlib-11557c?style=for-the-badge)](https://matplotlib.org/)
[![seaborn](https://img.shields.io/badge/seaborn-444876?style=for-the-badge&logo=graph&logoColor=white)](https://seaborn.pydata.org/)
[![tqdm](https://img.shields.io/badge/tqdm-FFC107?style=for-the-badge&logo=tqdm&logoColor=000000)](https://tqdm.github.io/)
[![pytest](https://img.shields.io/badge/pytest-%230A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/en/stable/)
[![ruff](https://img.shields.io/badge/ruff-D7FF64?style=for-the-badge&logo=ruff&logoColor=000000)](https://docs.astral.sh/ruff/)
