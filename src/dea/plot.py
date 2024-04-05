"""Plotting functions"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(
        window_length: np.ndarray,
        entropies: np.ndarray,
        x_interval: np.ndarray,
        slope: float,
        y_intercept: float,
        mu: float
    ) -> plt.axes:
    """Plot the slope of entropy vs window length, principal result of DEA"""
    fig, ax = plt.subplots()
    ax.plot(window_length, entropies, linestyle='', marker='.')
    ax.plot(
        x_interval,
        slope * np.log(x_interval) + y_intercept,
        color='k',
        label=f'$\\delta = {np.round(slope, 2)}$'
    )
    ax.plot([], [], linestyle='', label=f'$\\mu = {np.round(mu, 2)}$')
    plt.show(fig)

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
    ax.plot(
        mu1,
        delta,
        marker='o',
        fillstyle="none",
        markeredgewidth=2,
        linestyle='none',
        label=f'$\\mu$ = {np.round(mu1, 2)}'
    )
    ax.plot(
        mu2,
        delta,
        marker='o',
        fillstyle="none",
        markeredgewidth=2,
        linestyle='none',
        label=f'$\\mu$ = {np.round(mu2, 2)}'
    )
    ax.set_xticks(ticks=np.linspace(1, 4, 7))
    ax.set_yticks(ticks=np.linspace(0, 1, 5))
    ax.set_xlabel('$\\mu$')
    ax.set_ylabel('$\\delta$')
    ax.legend(loc=0)
    ax.grid(True)
    sns.despine(left=True, bottom=True)
    plt.show(fig)
