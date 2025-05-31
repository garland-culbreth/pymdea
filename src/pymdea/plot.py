"""Plotting functions."""

from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pymdea.core import DeaEngine


class DeaPlotter:
    """Plot DEA results."""

    def __init__(
        self: Self,
        model: DeaEngine,
        theme: None | str = None,
    ) -> Self:
        """Plot DEA results.

        Parameters
        ----------
        model : Self@DeaEngine
            Object containing the results of a DEA analysis to be plotted.
        theme : None | str, optional, default: None
            Must be either None or a string corresponding to a
            matplotlib.pyplot style.

        """
        if theme is not None:
            plt.style.use(style=theme)
        self.window_lengths = model.window_lengths
        self.entropies = model.entropies
        self.delta = model.fit_coefficients[0]
        self.y_intercept = model.fit_coefficients[1]
        self.mu1 = model.mu1
        self.mu2 = model.mu2
        self.mu3 = model.mu3

    def s_vs_l(self: Self, fig_width: int = 5, fig_height: int = 3) -> None:
        """Plot the slope of entropy vs window length.

        Parameters
        ----------
        fig_width : int, optional, default: 4
            Width, in inches, of the figure.
        fig_height : int, optional, default: 3
            Height, in inches, of the figure.

        """
        x_line = np.linspace(start=1, stop=np.max(self.window_lengths), num=3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout="constrained")
        ax.plot(
            self.window_lengths,
            self.entropies,
            linestyle="none",
            marker="o",
            markersize=3,
            fillstyle="none",
        )
        ax.plot(
            x_line,
            self.delta * np.log(x_line) + self.y_intercept,
            color="k",
            label=f"$\\delta = {np.round(self.delta, 3)}$",
        )
        ax.set_xscale("log")
        ax.set_xlabel("$\\ln(L)$")
        ax.set_ylabel("$S(L)$")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        sns.despine(trim=True)
        self.fig_s_vs_l = fig

    def mu_candidates(self: Self, fig_width: int = 5, fig_height: int = 3) -> None:
        """Plot the possible values of mu.

        Parameters
        ----------
        fig_width : int, optional, default: 4
            Width, in inches, of the figure.
        fig_height : int, optional, default: 3
            Height, in inches, of the figure.

        """
        x1 = np.linspace(1, 2, 100)
        x2 = np.linspace(2, 3, 100)
        x3 = np.linspace(3, 4, 100)
        y1 = x1 - 1
        y2 = 1 / (x2 - 1)
        y3 = np.full(100, 0.5)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout="constrained")
        ax.plot(x1, y1, color="k")
        ax.plot(x2, y2, color="k")
        ax.plot(x3, y3, color="k")
        ax.plot(
            self.mu1,
            self.delta,
            marker="o",
            fillstyle="none",
            markersize=5,
            markeredgewidth=2,
            linestyle="none",
            label=f"$\\mu$ = {np.round(self.mu1, 2)}",
        )
        ax.plot(
            self.mu2,
            self.delta,
            marker="o",
            fillstyle="none",
            markersize=5,
            markeredgewidth=2,
            linestyle="none",
            label=f"$\\mu$ = {np.round(self.mu2, 2)}",
        )
        ax.plot(
            self.mu3,
            self.delta,
            marker="o",
            fillstyle="none",
            markersize=5,
            markeredgewidth=2,
            linestyle="none",
            label=f"$\\mu$ = {np.round(self.mu3, 2)}",
        )
        ax.set_xticks(ticks=np.linspace(1, 4, 7))
        ax.set_yticks(ticks=np.linspace(0, 1, 5))
        ax.set_xlabel("$\\mu$")
        ax.set_ylabel("$\\delta$")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.grid(visible=True)
        sns.despine(left=True, bottom=True)
        self.fig_mu_candidates = fig
