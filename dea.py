
# Diffusion Entropy Analysis, with Stripes
#
# 2020-08-19 - Garland Culbreth
# Center for Nonlinear Science, University of North Texas.
#
# Repo:
# https://github.com/garland-culbreth/Diffusion-Entropy-Analysis
#
# For detailed function docstrings and other information, see dea_demo.ipynb in
# the repo.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fbm import fbm, fgn
from tqdm import tqdm

plt.style.use('ggplot')


def sample_data(length):
    """Generates an array of sample data."""
    # hurst = 0.7
    # fGnSample = fgn(length, hurst)
    # fBmSample = fbm(length, hurst)
    np.random.seed(0)
    random_steps = np.random.choice([-1, 1], length)
    random_steps[0] = 0  # always start from 0
    random_walk = np.cumsum(random_steps)
    return random_walk


def apply_stripes(data, stripes, show_plot):
    """
    Rounds `data` to `stripes` evenly spaced intervals.

    Parameters
    ----------
    data : array_like
        Time-series data to be examined.
    stripes : int
        Number of stripes to apply. 
    show_plot : int
        If 1, show data plot with overlaid stripes.

    Returns
    ----------
    rounded_data : ndarray
        data rounded to stripes number of equally spaced intervals.
    """
    if min(data) <= 0:
        data = data + abs(min(data))
    elif min(data) > 0:
        data = data - abs(min(data))
    max_data = max(data)
    min_data = min(data)
    data_width = abs(max_data - min_data)
    stripe_size = data_width / stripes
    rounded_data = data / stripe_size
    if show_plot == 1:
        lines = np.linspace(min_data, max_data, num=stripes)
        plt.figure(figsize=(6, 5))
        plt.plot(data)
        plt.hlines(y=lines, xmin=0, xmax=len(data))
        plt.xlabel('t')
        plt.ylabel('Data(t)')
        plt.title('Data with stripes')
        plt.show()
    return rounded_data


def find_events(series):
    """Records an event (1) when `series` changes value."""
    """Records an event (1) when `series` changes value."""
    events = []
    for i in range(1, len(series)):
        if (series[i] < np.floor(series[i-1])+1 and 
            series[i] > np.ceil(series[i-1])-1):
            events.append(0)
        else:
            events.append(1)
    np.append(events, 0)
    return events


def make_trajectory(events):
    """Constructs diffusion trajectory from events."""
    trajectory = np.cumsum(events)
    return trajectory


def entropy(trajectory):
    """
    Calculates the Shannon Entropy of the diffusion trajectory.

    Generates a range of window lengths L. Steps each one along 
    `trajectory` and computes the displacement of `trajectory` over 
    each window position. Bins these displacements, and divides by the 
    sum of all bins to make the probability distribution `P`. Puts `P` 
    into the equation for Shannon Entropy to get S(L). Repeats for all 
    L in range `WindowLengths`.

    Parameters
    ----------
    trajectory : array_like
        Diffusion trajectory. Constructed by make_trajectory.

    Returns
    ----------
    S : ndarray
        Shannon Entropy values, S(L).
    window_lengths : ndarray
        Window lengths, L. 

    Notes
    ----------
    `tqdm()` makes the progress bar appear.
    """
    S = []
    window_lengths = np.arange(1, int(0.25*len(trajectory)), 1)
    for L in tqdm(window_lengths):
        window_starts = np.arange(0, len(trajectory)-L, 1)
        window_ends = np.arange(L, len(trajectory), 1)
        displacements = trajectory[window_ends] - trajectory[window_starts]
        bin_counts = np.bincount(displacements)
        bin_counts = bin_counts[bin_counts != 0]
        P = bin_counts / np.sum(bin_counts)
        S.append(-np.sum(P * np.log(P)))
    return S, window_lengths


def no_stripe_entropy(trajectory):
    """
    Calculates the Shannon Entropy of the diffusion trajectory.

    Oridnary DEA function by David Lambert and Jacob Baxley,
    vectorized by Garland Culbreth.
    Generates a range of window lengths L. Steps each one along 
    `trajectory` and computes the displacement of `trajectory` 
    over each window position. Bins these displacements, and divides 
    by the sum of all bins to make the probability distribution `p`. 
    Puts `p` into the equation for Shannon Entropy to get s(L).
    Repeats for all L in range `WindowLengths`.

    Parameters
    ----------
    trajectory : array_like
        Diffusion trajectory. FOR NO STRIPES JUST PASS THE DATA SERIES.

    Returns
    ----------
    S : ndarray
        Shannon Entropy values, S(L).
    window_lengths : ndarray
        Window lengths, L.

    Notes
    ----------
    `tqdm()` makes the progress bar appear.
    """
    window_lengths = np.arange(1, int(0.25*len(data)), 1)
    S = []
    for L in tqdm(window_lengths):
        window_starts = np.arange(0, len(trajectory)-L, 1)
        window_ends = np.arange(L, len(trajectory), 1)
        traj = trajectory[window_starts] - trajectory[window_ends]
        # This part does the actual DEA computations
        counts, bin_edge = np.histogram(traj, bins='doane')  # doane least bad
        counts = np.array(counts[counts != 0])
        binsize = bin_edge[1] - bin_edge[0]
        P = counts / sum(counts)
        S.append(-sum(P*np.log(P)) + np.log(binsize))
    return S, window_lengths


def get_scaling(S, L, start, stop):
    """
    Calculates scaling.

    Calculates the scaling of the time-series by performing a 
    least-squares linear fit over S(l) and ln(l).

    Parameters
    ----------
    S : array_like
        Shannon Entropy values. 
    L : array_like
        Window Lengths. 
    start : int
        Index at which to start the fit slice.
    stop : int
        Index at which to stop the fit slice.

    Returns
    ----------
    L_slice : ndarray 
        The slice of window lengths L.
    fit_coeffs : ndarray
        Slope and intercept of the fit. 

    Notes
    ----------
    Least-squares linear fits on log scale data have issues, see 
    doi:10.1371/journal.pone.0085777.
    Making a version that uses the `powerlaw` package instead would 
    be better...
    """
    S_slice = S[start:stop]
    L_slice = L[start:stop]
    fit_coeffs = np.polyfit(np.log(L_slice), S_slice, 1)
    return L_slice, fit_coeffs


def get_mu(delta):
    """
    Calculates mu (powerlaw index of inter-event time distribution).

    Parameters
    ----------
    delta : float
        Scaling of the time-series process. 

    Returns
    ----------
    mu : float
        Complexity parameter. Powerlaw index for inter-event time 
        distribution.
    """
    mu = 1 + (1 / delta)
    if mu > 3:
        mu = 1 + delta
    return mu


def dea_no_stripes(data, start, stop):
    """
    Applies DEA without the stripes refinement.

    Original DEA. Takes the original time series as the diffusion 
    trajectory. 

    Parameters
    ----------
    data : array_like
        Time-series to be analysed.
    asym : bool
        True or False, whether to use asymmetric diffusion.
    start : int
        Array index at which to start linear fit.
    stop : int 
        Array index at which to stop linear fit.

    Returns
    ----------
    figure 
        A figure plotting S(l) vs. ln(l), overlaid with the fit 
        line, labelled with the scaling and mu values.
    """
    S, L = no_stripe_entropy(data)
    fit = get_scaling(S, L, start, stop)
    mu = get_mu(fit[1][0])

    fig = plt.figure(figsize = (6, 5))
    plt.plot(L, S, linestyle='', marker='.')
    plt.plot(fit[0], fit[1][0] * np.log(fit[0]) + fit[1][1], color='k',
             label='$\delta = {}$'.format(np.round(fit[1][0], 2)))
    plt.plot([], [], linestyle='', 
             label='$\mu = {}$'.format(np.round(mu, 2)))
    plt.xscale('log')
    plt.xlabel('$ln(l)$')
    plt.ylabel('$S(l)$')
    plt.legend(loc=0)
    # plt.show()
    return fig


def dea_with_stripes(data, stripes, start, stop, data_plot):
    """
    Applies DEA with the stripes refinement.

    Runs a sequence of functions to apply stripes and then perform 
    DEA on the data series. 

    Parameters
    ----------
    data : array_like
        Time-series to be analysed.
    stripes : int
        Number of stripes to be applied to the data.
    start : int
        Array index at which to start linear fit.
    stop : int 
        Array index at which to stop linear fit.
    data_plot : int
        If 1, show data plot with overlaid stripes.

    Returns
    ----------
    fig : figure 
        A figure plotting S(l) vs. ln(l), overlaid with the fit line, 
        labelled with the scaling and mu values.
    """
    rounded_data = apply_stripes(data, stripes, data_plot)
    event_array = find_events(rounded_data)
    diffusion_trajectory = make_trajectory(event_array)
    S, L = entropy(diffusion_trajectory)
    fit = get_scaling(S, L, start, stop)
    mu = get_mu(fit[1][0])

    fig = plt.figure(figsize=(6, 5))
    plt.plot(L, S, linestyle='', marker='.')
    plt.plot(fit[0], fit[1][0]*np.log(fit[0]) + fit[1][1],
             color='k', label='$\\delta = $'+str(np.round(fit[1][0], 2)))
    plt.plot([], [], linestyle='', label='$\\mu = $'+str(np.round(mu, 2)))
    plt.xscale('log')
    plt.xlabel('$ln(L)$')
    plt.ylabel('$S(L)$')
    plt.legend(loc=0)
    return fig


### ----WORK HERE---- ###
data = sample_data(20000)
number_of_stripes = 40  # needs to be at least 2
fit_start = 50
fit_stop = 400
show_data_plot = 0  # set to 1 to see plot of data with stripes

result = dea_with_stripes(data,
                          number_of_stripes,
                          fit_start,
                          fit_stop,
                          show_data_plot)
plt.show()
