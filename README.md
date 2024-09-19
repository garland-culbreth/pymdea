# Diffusion entropy analysis

[![pytest status](https://img.shields.io/github/actions/workflow/status/garland-culbreth/pymdea/pytest.yml?style=flat-square&logo=github&&label=tests&logoColor=white&labelColor=393f46&color=46954a)](https://github.com/garland-culbreth/pymdea/actions/workflows/pytest.yml)
[![mkdocs status](https://img.shields.io/github/actions/workflow/status/garland-culbreth/pymdea/mkdocs.yml?style=flat-square&logo=github&logoColor=white&label=docs&labelColor=393f46&color=46954a)](https://github.com/garland-culbreth/pymdea/actions/workflows/mkdocs.yml)
[![python versions](https://img.shields.io/badge/3.11_%7C_3.12-3776AB?style=flat-square&logo=python&logoColor=ffffff&label=python&labelColor=393f46)](https://github.com/garland-culbreth/pymdea)
[![pypi version](https://img.shields.io/pypi/v/pymdea?style=flat-square&logo=pypi&logoColor=ffffff&labelColor=393f46)](https://pypi.org/project/pymdea/)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square&labelColor=393f46)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=flat-square&labelColor=393f46)](https://github.com/astral-sh/uv)

Diffusion Entropy Analysis is a time-series analysis method for detecting temporal scaling in a data set, such as particle motion, a seismograph, or an electroencephalograph signal. Diffusion Entropy Analysis converts a timeseries into a diffusion trajectory and uses the entropy of this trajectory to measure the temporal scaling in the data. This is accomplished by moving a window along the trajectory, then using the relationship between the natural logarithm of the length of the window and the Shannon entropy to extract the scaling of the time-series process.

For further details about the method and how it works, please see [Culbreth, G., Baxley, J. and Lambert, D., 2023. Detecting temporal scaling with modified diffusion entropy analysis. _arXiv preprint arXiv:2311.11453_](https://arxiv.org/abs/2311.11453).

## Installation and use

The pymdea package is available on pypi and can be installed with pip:
```bash
pip install pymdea
```
pymdea can also be installed with [uv](https://docs.astral.sh/uv/#python-management)
```bash
uv add pymdea
```

A user guide is available in the [documentation](https://garland-culbreth.github.io/pymdea/).

## Built with

[![numpy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scipy](https://img.shields.io/badge/scipy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![polars](https://img.shields.io/badge/polars-CD792C?style=for-the-badge&logo=polars&logoColor=white)](https://pola.rs/)
[![matplotlib](https://img.shields.io/badge/matplotlib-11557c?style=for-the-badge)](https://matplotlib.org/)
[![seaborn](https://img.shields.io/badge/seaborn-444876?style=for-the-badge&logo=graph&logoColor=white)](https://seaborn.pydata.org/)
[![tqdm](https://img.shields.io/badge/tqdm-FFC107?style=for-the-badge&logo=tqdm&logoColor=000000)](https://tqdm.github.io/)
[![pytest](https://img.shields.io/badge/pytest-%230A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/en/stable/)
[![ruff](https://img.shields.io/badge/ruff-D7FF64?style=for-the-badge&logo=ruff&logoColor=000000)](https://docs.astral.sh/ruff/)
[![material for mkdocs](https://img.shields.io/badge/material_for_mkdocs-%23526CFE?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)
[![mkdocstrings](https://img.shields.io/badge/mkdocstrings-%23526CFE?style=for-the-badge)](https://mkdocstrings.github.io/)
