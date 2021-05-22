# Diffusion-Entropy-Analysis

Garland Culbreth, Jacob Baxley, & David Lambert  
Center for Nonlinear Science, University of North Texas


# Use

* dea.ipynb is for working with real data.
* dea_demo.ipynb is designed to demonstrate the method.

The Jupyter Notebooks contain detailed use instructions inside in markdown cells.

You will need to have an up-to-date Python installation. We recommend [Anaconda](https://www.anaconda.com/products/individual) for this, as it also includes all dependencies needed for Diffusion Entropy Analysis.


# About

Diffusion Entropy Analysis is a time-series analysis method for detecting temporal scaling in a data set, such as particle motion, a seismograph, or an electroencephalograph signal. Diffusion Entropy Analysis converts a timeseries into a diffusion trajectory and uses the entropy of this trajectory to measure the temporal scaling in the data. This is accomplished by moving a window along the trajectory, then using the relationship between the natural logarithm of the length of the window and the Shannon entropy to extract the scaling of the time-series process.
