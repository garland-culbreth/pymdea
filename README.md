# Diffusion-Entropy-Analysis

Garland Culbreth, Jacob Baxley, & David Lambert  
Center for Nonlinear Science, University of North Texas

# About

Diffusion Entropy Analysis is a time-series analysis method for detecting temporal scaling in a data set, such as particle motion, a seismograph, or an electroencephalograph signal. Diffusion Entropy Analysis converts a timeseries into a diffusion trajectory and uses the entropy of this trajectory to measure the temporal scaling in the data. This is accomplished by moving a window along the trajectory, then using the relationship between the natural logarithm of the length of the window and the Shannon entropy to extract the scaling of the time-series process.

# Use

You will need to have an up-to-date Python installation, e.g., through [Anaconda](https://www.anaconda.com/products/individual). The Jupyter Notebooks contain detailed use instructions inside in markdown cells.

dea.ipynb is for working with real data.  
dea_demo.ipynb is designed to demonstrate the method.

# Dependencies

[NumPy](https://numpy.org/doc/stable/index.html)
[Scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
[Polars](https://www.pola.rs/)
[Seaborn](https://seaborn.pydata.org/)
[Matplotlib](https://matplotlib.org/)
[tqdm](https://tqdm.github.io/)

