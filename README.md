# Diffusion-Entropy-Analysis

Diffusion Entropy Analysis is a time-series analysis method for detecting temporal scaling in a data set, such as particle motion, a seismograph, or an electroencephalograph signal. Diffusion Entropy Analysis converts a timeseries into a diffusion trajectory and uses the entropy of this trajectory to measure the temporal scaling in the data. This is accomplished by moving a window along the trajectory, then using the relationship between the natural logarithm of the length of the window and the Shannon entropy to extract the scaling of the time-series process.

For an in-depth introduction to the method and how it works, please see [Culbreth, G., Baxley, J. and Lambert, D., 2023. Detecting temporal scaling with modified diffusion entropy analysis. _arXiv preprint arXiv:2311.11453_](https://doi.org/10.48550/arXiv.2311.11453).

## Installation

You will need to have an up-to-date Python installation, e.g., through [Anaconda](https://www.anaconda.com/products/individual). Once you have a working Python install, clone this repository into a directory from which you can work with the files.

## Use

Once you have cloned this repository to your local machine, use the `dea.ipynb` file to prepare your data and run the diffusion entropy analysis. A user guide and quick reference are provided on [the wiki](https://github.com/garland-culbreth/Diffusion-Entropy-Analysis/wiki).

### dea.ipynb
This Jupyter notebook is for working with real data. It imports functions from the `dea.py` module to make code cells as simple as possible, and contains markdown discussing how to use the method and interpret the results.

### dea/core.py
This file is a Python module containing all the functions which perform the core of diffusion entropy analysis.

### dea/util.py
This file is a Python module containing utility helper functions.

### dea/plot.py
This file is a Python module containing plotting helper functions.

## Dependencies

[NumPy](https://numpy.org/doc/stable/index.html)
[Scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
[Polars](https://www.pola.rs/)
[Seaborn](https://seaborn.pydata.org/)
[Matplotlib](https://matplotlib.org/)
[tqdm](https://tqdm.github.io/)
[ipywidgets](https://ipywidgets.readthedocs.io/en/stable/)

