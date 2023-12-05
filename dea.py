"""Diffusion entropy analysis module

A collection of functions which run the diffusion entropy
analysis algorithm for temporal complexity detection.
"""
import os
from typing import Union
import time
import numpy as np
from scipy import stats
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def make_sample_data(length: int, seed: float = time.time) -> np.ndarray:
    """Generates an array of sample data."""
    np.random.seed(seed)  # for baseline consistency 1010
    random_steps = np.random.choice([-1, 1], length)
    random_steps[0] = 0  # always start from 0
    random_walk = np.cumsum(random_steps)
    return random_walk


def get_data(filepath: str) -> pl.DataFrame:
    """Convenience function for reading input data
    
    Parameters
    ----------
    filepath : str
        System path to a file containing data.
        Must include the full file name, including the extension.
        Example: "/example/path/to/file.csv"
    
    Returns
    -------
    data : DataFrame
        A polars DataFrame containing the data.
    
    Raises
    ------
    ValueError
        If filepath points to a file of type other than
        CSV. Support for more types of files is a work in
        progress.
    """
    filetype = os.path.splitext(filepath)[1]
    if filetype not in [".csv"]:
        raise ValueError("Parameter 'filetype' must be one of: ['.csv'].")
    if filetype == ".csv":
        data = pl.scan_csv(filepath)
    return data


def apply_stripes(data: Union[np.ndarray, pl.Series],
                  stripes: int,
                  show_data_plot: bool = False) -> np.ndarray:
    """
    Rounds `data` to `stripes` evenly spaced intervals.

    Parameters
    ----------
    data : array_like
        Time-series data to be examined.
    stripes : int
        Number of stripes to apply. 
    show_data_plot : bool
        If True, show data plot with overlaid stripes.

    Returns
    ----------
    rounded_data : ndarray
        `data` rounded to `stripes` number of equally spaced intervals.
    """
    if stripes < 2:
        raise ValueError("Parameter 'stripes' must be greater than 2.")
    if show_data_plot == True:
        lines = np.linspace(min(data), max(data), num=stripes)
        plt.figure(figsize=(5, 4))
        plt.plot(data)
        plt.hlines(
            y=lines,
            xmin=0,
            xmax=len(data),
            colors='0.3',
            linewidths=1,
            alpha=0.4
        )
        plt.xlabel('t')
        plt.ylabel('Data(t)')
        plt.title('Data with stripes')
        sns.despine()
        plt.show()

    if min(data) <= 0:
        data = data + abs(min(data))
    elif min(data) > 0:
        data = data - abs(min(data))
    max_data = max(data)
    min_data = min(data)
    data_width = abs(max_data - min_data)
    stripe_size = data_width / stripes
    rounded_data = data / stripe_size
    return rounded_data


def get_events(series: Union[np.ndarray, pl.Series]) -> np.ndarray:
    """Records an event (1) when `series` changes value."""
    events = []
    for i in range(1, len(series)):
        if (series[i] < np.floor(series[i-1])+1 and
            series[i] > np.ceil(series[i-1])-1):
            # if both true, no crossing
            events.append(0)
        else:
            events.append(1)
    np.append(events, 0)
    return events


def make_trajectory(events: np.ndarray) -> list[float]:
    """Constructs diffusion trajectory from events."""
    trajectory = np.cumsum(events)
    return trajectory


def get_entropy(trajectory: np.ndarray) -> list[np.ndarray]:
    """
    Calculates the Shannon Entropy of the diffusion trajectory.

    Generates a range of window lengths L. Steps each one along 
    'trajectory' and computes the displacement of 'trajectory' 
    over each window position. Bins these displacements, and divides 
    by the sum of all bins to make the probability distribution 'p'. 
    Puts 'p' into the equation for Shannon Entropy to get S(L).
    Repeats for all L in range 'window_lengths'.

    Parameters
    ----------
    trajectory : array_like
        Diffusion trajectory. Constructed by make_trajectory.

    Returns
    ----------
    entropies : ndarray
        Shannon Entropy values, S(L).
    window_lengths : ndarray
        Window lengths, L. 

    Notes
    ----------
    'tqdm(...)' makes the progress bar appear.
    """
    entropies = []
    window_lengths = np.arange(1, int(0.25*len(trajectory)), 1)
    for window_length in tqdm(window_lengths):
        window_starts = np.arange(0, len(trajectory)-window_length, 1)
        window_ends = np.arange(window_length, len(trajectory), 1)
        displacements = trajectory[window_ends] - trajectory[window_starts]
        counts, bin_edge = np.histogram(displacements, bins='doane')
        counts = np.array(counts[counts != 0])
        binsize = bin_edge[1] - bin_edge[0]
        distribution = counts / sum(counts)
        entropies.append(-sum(distribution*np.log(distribution))
                         + np.log(binsize))
    return entropies, window_lengths


def get_no_stripe_entropy(trajectory: np.ndarray) -> list[np.ndarray]:
    """
    Calculates the Shannon Entropy of the diffusion trajectory.

    Generates a range of window lengths window_length. Steps each one along 
    'trajectory' and computes the displacement of 'trajectory' 
    over each window position. Bins these displacements, and divides 
    by the sum of all bins to make the probability distribution 'p'. 
    Puts 'p' into the equation for Shannon Entropy to get S(L).
    Repeats for all L in range 'window_lengths'.

    Parameters
    ----------
    trajectory : array_like
        Diffusion trajectory. FOR NO STRIPES JUST PASS THE DATA SERIES.

    Returns
    ----------
    entropies : ndarray
        Shannon Entropy values, S(L).
    window_lengths : ndarray
        Window lengths, L.

    Notes
    ----------
    `tqdm()` makes the progress bar appear.
    """
    window_lengths = np.arange(1, int(0.25*len(trajectory)), 1)
    entropies = []
    for window_length in tqdm(window_lengths):
        window_starts = np.arange(0, len(trajectory)-window_length, 1)
        window_ends = np.arange(window_length, len(trajectory), 1)
        traj = trajectory[window_starts] - trajectory[window_ends]
        # Use bins='doane'; least bad for nongaussian
        counts, bin_edge = np.histogram(traj, bins='doane')
        counts = np.array(counts[counts != 0])
        binsize = bin_edge[1] - bin_edge[0]
        distribution = counts / sum(counts)
        entropies.append(-sum(distribution*np.log(distribution))
                         + np.log(binsize))
    return entropies, window_lengths


def get_scaling(entropies: np.ndarray,
                window_length: np.ndarray,
                start: list[float],
                stop: int,
                fit_method: str = "siegel") -> list[np.ndarray]:
    """
    Calculates scaling.
    
    Calculates the scaling of the time-series by performing a 
    least-squares linear fit over S(l) and ln(l).

    Parameters
    ----------
    entropies : array_like
        Shannon Entropy values. 
    window_length : array_like
        Window Lengths. 
    start : int
        Index at which to start the fit slice.
    stop : int
        Index at which to stop the fit slice.
    fit_method : str {"siegel", "theilsen", "ls"}, optional
        Linear fit method to use. By default "siegel"

    Returns
    -------
    fit_slice_L : ndarray 
        The slice of window lengths window_length.
    coefficients : ndarray
        Slope and intercept of the fit. 

    Notes
    -----
    Prefer the siegel or theilsen methods. Least squares linear
    fits can introduce bias when done over log-scale data, see
    Clauset, A., Shalizi, C.R. and Newman, M.E., 2009. Power-law
    distributions in empirical data. SIAM review, 51(4), pp.661-703.
    https://doi.org/10.1137/070710111.
    https://arxiv.org/pdf/0706.1062.pdf.
    """
    if fit_method not in ["siegel", "theilsen", "ls"]:
        raise ValueError(
            "Parameter 'method' must be one of: ['siegel', 'theilsen', 'ls']."
        )
    s_slice = entropies[start:stop]
    length_slice = window_length[start:stop]
    if fit_method == "ls":
        coefficients = np.polyfit(np.log(length_slice), s_slice, 1)
    if fit_method == "theilsen":
        coefficients = stats.theilslopes(s_slice, np.log(length_slice))
    if fit_method == "siegel":
        coefficients = stats.siegelslopes(s_slice, np.log(length_slice))
    return length_slice, coefficients


def get_mu(delta: float) -> list[float]:
    """
    Calculates the mu.

    Parameters
    ----------
    delta : float
        Scaling of the time-series process. 

    Returns
    ----------
    mu : float
        Complexity parameter. Powerlaw index for inter-event 
        time distribution.
    Notes
    ----------
    mu is calculated by both rules. later both are plotted
    against the line relating delta and mu, to hopefully
    let users graphically determine the correct mu.
    """
    mu1 = 1 + delta
    mu2 = 1 + (1 / delta)
    return mu1, mu2


def plot_results(window_length: np.ndarray,
                 entropies: np.ndarray,
                 x_interval: np.ndarray,
                 slope: float,
                 y_intercept: float,
                 mu: float) -> plt.axes:
    fig, ax = plt.subplots()
    ax.plot(window_length, entropies, linestyle='', marker='.')
    ax.plot(
        x_interval,
        slope * np.log(x_interval) + y_intercept,
        color='k',
        label=f'$\\delta = {np.round(slope, 2)}$'
    )
    ax.plot([], [], linestyle='', label=f'$\\mu = {np.round(mu, 2)}$')
    return ax


def plot_mu_candidates(delta: float, mu1: float, mu2: float) -> None:
    """Plots the possible values of mu."""
    x1 = np.linspace(1, 2, 100)
    x2 = np.linspace(2, 3, 100)
    x3 = np.linspace(3, 4, 100)
    y1 = x1 - 1
    y2 = 1 / (x2 - 1)
    y3 = np.full(100, 0.5)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x1, y1, color='k')
    ax.plot(x2, y2, color='k')
    ax.plot(x3, y3, color='k')
    ax.plot(mu1, delta, marker='o', label=f'$\\mu$ = {np.round(mu1, 2)}')
    ax.plot(mu2, delta, marker='o', label=f'$\\mu$ = {np.round(mu2, 2)}')
    ax.set_xticks(ticks=np.linspace(1, 4, 7))
    ax.set_yticks(ticks=np.linspace(0, 1, 5))
    ax.set_xlabel('$\\mu$')
    ax.set_ylabel('$\\delta$')
    ax.legend(loc=0)
    ax.grid(True)
    sns.despine(left=True, bottom=True)
    plt.show(fig)
    return None


def run_dea_no_stripes(data: Union[np.ndarray, pl.Series],
                       fit_start: int, fit_stop: int,
                       fit_method: str = "siegel") -> None:
    """
    Applies DEA without the stripes refinement.

    Original DEA. Takes the original time series as the diffusion 
    trajectory.

    Parameters
    ----------
    data : array_like
        Time-series to be analysed.
    start : int
        Array index at which to start linear fit.
    stop : int 
        Array index at which to stop linear fit.
    fit_method : str {"siegel", "theilsen", "ls"}, optional
        Linear fit method to use. By default "siegel"

    Returns
    ----------
    figure 
        A figure plotting S(l) vs. ln(l), overlaid with the fit 
        line, labelled with the scaling and mu values.
    """
    print("Beginning DEA without stripes.")
    entropies, window_length = get_no_stripe_entropy(data)
    fit = get_scaling(entropies, window_length, fit_start, fit_stop, fit_method)
    mu = get_mu(fit[1][0])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(window_length, entropies, linestyle='', marker='.', alpha=0.5)
    ax.plot(
        fit[0],
        fit[1][0] * np.log(fit[0]) + fit[1][1],
        color='k',
        label=f'$\\delta = {np.round(fit[1][0], 3)}$'
    )
    ax.plot(
        [], [], linestyle='', label=f'$\\mu = {np.round(mu, 3)}$')
    ax.xscale('log')
    ax.xlabel('$ln(l)$')
    ax.ylabel('$S(l)$')
    ax.legend(loc=0)
    sns.despine()
    plt.show(fig)
    print("DEA without stripes complete.")
    return None


def run_dea_with_stripes(data: Union[np.ndarray, pl.Series],
                         number_of_stripes: int,
                         fit_start: int,
                         fit_stop: int,
                         fit_method: str = "siegel",
                         show_data_plot: bool = False) -> None:
    """
    Applies DEA with the stripes refinement.

    Runs a sequence of functions to apply stripes and then 
    perform DEA on the data series. 

    Parameters
    ----------
    data : array_like
        Time-series to be analysed.
    number_of_stripes : int
        Number of stripes to be applied to the data.
    fit_start : int
        Array index at which to start linear fit.
    fit_stop : int 
        Array index at which to stop linear fit.
    fit_method : str {"siegel", "theilsen", "ls"}, optional
        Linear fit method to use. By default "siegel"
    show_data_plot : bool
        If True, show data plot with overlaid stripes.

    Returns
    ----------
    fig : figure 
        A figure plotting S(l) vs. ln(l), overlaid with the fit 
        line, labelled with the scaling and mu values.
    """
    print("Beginning DEA with stripes.")
    rounded_data = apply_stripes(data, number_of_stripes, show_data_plot)
    event_array = get_events(rounded_data)
    diffusion_trajectory = make_trajectory(event_array)
    entropies, window_length = get_entropy(diffusion_trajectory)
    fit = get_scaling(entropies, window_length, fit_start, fit_stop, fit_method)
    mu = get_mu(fit[1][0])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(window_length, entropies, linestyle='', marker='.', alpha=0.5)
    ax.plot(
        fit[0],
        fit[1][0] * np.log(fit[0]) + fit[1][1],
        color='k',
        label=f'$\\delta = {np.round(fit[1][0], 3)}$'
    )
    ax.set_xscale('log')
    ax.set_xlabel('$ln(L)$')
    ax.set_ylabel('$S(L)$')
    ax.legend(loc=0)
    sns.despine()
    plt.show(fig)

    plot_mu_candidates(fit[1][0], mu[0], mu[1])
    print("DEA with stripes complete.")
    return None
