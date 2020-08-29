"""
Diffusion Entropy Analysis, with Stripes  

Garland Culbreth - Center for Nonlinear Science, University of North Texas.  

Written 2020-08-19. 
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fbm import fbm, fgn
from tqdm import tqdm

plt.style.use('ggplot')


length = 20000
# Hurst = 0.7
# fGnSample = fgn(Length, Hurst)
# fBmSample = fbm(Length, Hurst)

np.random.seed(0)
random_steps = np.random.choice([-1, 1], length)
random_steps[0] = 0  # always start from 0
random_walk = np.cumsum(random_steps)

# plt.figure(figsize = (6, 5))
# plt.plot(random_walk)
# plt.xlabel('t')
# plt.ylabel('x(t)')
# plt.title('Sample Data')
# plt.show()


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

    Notes
    ----------
    The reason I use `floor()` and `ceil()` rather than just `round()`
    is because `round()` puts the rounding thresholds at half-integers, 
    while `floor()` and `ceil()` put them exactly at the integers. 
    This makes no difference to the analysis, it just bothers me.
    """
    max_data = max(data)
    min_data = min(data)
    data_width = abs(max_data - min_data)
    stripe_size = data_width / stripes
    rounded_data = []
    for i in range(len(data)):
        if data[i] >= 0:
            rounded_data.append(np.floor(data[i] / stripe_size))
        elif data[i] < 0:
            rounded_data.append(np.ceil(data[i] / stripe_size))
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
    """
    Records an event when `series` changes value. 

    Parameters
    ----------
    series : array_like
        Data series rounded to the stripe intervals.

    Returns
    ----------
    events : ndarray
        1 if event occured at that time index, 0 if not.
    """
    events = []
    for i in range(len(series)-1):
        if series[i] != series[i+1]:
            events.append(1)
        else:
            events.append(0)
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
    sum of all bins to make the probability distribution `p`. Puts `p` 
    into the equation for Shannon Entropy to get s(L). Repeats for all 
    L in range `WindowLengths`.

    Parameters
    ----------
    trajectory : array_like
        Diffusion trajectory. Constructed by make_trajectory.

    Returns
    ----------
    s : ndarray
        Shannon Entropy values, S(L).
    window_lengths : ndarray
        Window lengths, L. 

    Notes
    ----------
    `tqdm()` makes the progress bar appear.
    """
    s = []
    window_lengths = np.arange(1, int(0.25*len(trajectory)), 1)
    for L in tqdm(window_lengths):
        window_starts = np.arange(0, len(trajectory)-L, 1)
        window_ends = np.arange(L, len(trajectory), 1)
        displacements = trajectory[window_ends] - trajectory[window_starts]
        bin_counts = np.bincount(displacements)
        bin_counts = bin_counts[bin_counts != 0]
        p = bin_counts / np.sum(bin_counts)
        s.append(-np.sum(p * np.log(p)))
    return s, window_lengths


def get_scaling(s, L, start, stop):
    """
    Calculates scaling.

    Calculates the scaling of the time-series by performing a 
    least-squares linear fit over S(l) and ln(l).

    Parameters
    ----------
    s : array_like
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
    coefficients : ndarray
        Slope and intercept of the fit. 

    Notes
    ----------
    Least-squares linear fits on log scale data have issues, see 
    doi:10.1371/journal.pone.0085777.
    Making a version that uses the `powerlaw` package instead would 
    be better...
    """
    s_slice = s[start:stop]
    L_slice = L[start:stop]
    coefficients = np.polyfit(np.log(L_slice), s_slice, 1)
    return L_slice, coefficients


def get_mu(delta):
    """
    Calculates the mu.

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

    !!! WIP !!!
    Takes signum function of your data, then runs DEA on the resulting 
    rounded series. 

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

    Returns
    ----------
    figure 
        A figure plotting S(l) vs. ln(l), overlaid with the fit line, 
        labelled with the scaling and mu values.
    """
    rounded_data = np.sign(data)
    event_array = find_events(rounded_data)
    diffusion_trajectory = make_trajectory(event_array)
    s, L = entropy(diffusion_trajectory)
    fit = get_scaling(s, L, start, stop)
    mu = get_mu(fit[1][0])

    fig = plt.figure(figsize=(6, 5))
    plt.plot(L, s, linestyle='', marker='.')
    plt.plot(fit[0], fit[1][0] * np.log(fit[0]) + fit[1][1],
             label='$\\delta = {}$'.format(np.round(fit[1][0], 3)))
    plt.plot([], [], linestyle='',
             label='$\\mu = {}$'.format(np.round(mu, 3)))
    plt.xscale('log')
    plt.xlabel('$ln(l)$')
    plt.ylabel('$S(l)$')
    plt.legend(loc=0)
    plt.show()
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
    s, L = entropy(diffusion_trajectory)
    fit = get_scaling(s, L, start, stop)
    mu = get_mu(fit[1][0])

    fig = plt.figure(figsize=(6, 5))
    plt.plot(L, s, linestyle='', marker='.')
    plt.plot(fit[0], fit[1][0] * np.log(fit[0]) + fit[1][1],
             label='$\\delta = {}$'.format(np.round(fit[1][0], 3)))
    plt.plot([], [], linestyle='',
             label='$\\mu = {}$'.format(np.round(mu, 3)))
    plt.xscale('log')
    plt.xlabel('$ln(L)$')
    plt.ylabel('$S(L)$')
    plt.legend(loc=0)
    plt.show()
    return fig


### ----WORK HERE---- ###
data = random_walk
number_of_stripes = 40  # needs to be at least 2
fit_start = 40
fit_stop = 800
show_data_plot = 0  # set to 1 to see plot of data with stripes

result = dea_with_stripes(data,
                          number_of_stripes,
                          fit_start,
                          fit_stop,
                          show_data_plot)
