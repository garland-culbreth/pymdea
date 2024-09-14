"""Plotting functions."""

from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pymdea.core import DeaEngine


class DeaPlotter:
    """Plot DEA results."""

    def __init__(
        self: Self,
        model: DeaEngine,
        theme: Literal["ticks", "whitegrid", "darkgrid"] = "ticks",
        colors: Literal["muted", "deep", "Set2", "tab10"] = "muted",
    ) -> Self:
        """Plot DEA results.

        Parameters
        ----------
        model : Self@DeaEngine
            Object containing the results of a DEA analysis to be plotted.
        theme : str {"ticks", "whitegrid", "darkgrid"}, optional, default: "ticks"
            Name of a seaborn style. Passed through to
            seaborn.set_theme().
        colors : str {"muted", "deep", "Set2", "tab10"}, optional, default: "muted"
            Name of a seaborn or matplotlib color palette. Passed
            through to seaborn.set_theme().

        """
        sns.set_theme(context="notebook", style=theme, palette=colors)
        self.window_lengths = model.window_lengths
        self.entropies = model.entropies
        self.delta = model.fit_coefficients[0]
        self.y_intercept = model.fit_coefficients[1]
        self.mu1 = model.mu1
        self.mu2 = model.mu2

    def s_vs_l(self: Self, fig_width: int = 4, fig_height: int = 3) -> None:
        """Plot the slope of entropy vs window length.

        Parameters
        ----------
        fig_width : int, optional, default: 4
            Width, in inches, of the figure.
        fig_height : int, optional, default: 3
            Height, in inches, of the figure.

        """
        x_line = np.linspace(start=1, stop=np.max(self.window_lengths), num=3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
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
        ax.legend(loc=0)
        sns.despine(trim=True)
        plt.show(fig)

    def mu_candidates(self: Self, fig_width: int = 4, fig_height: int = 3) -> None:
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

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
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
        ax.set_xticks(ticks=np.linspace(1, 4, 7))
        ax.set_yticks(ticks=np.linspace(0, 1, 5))
        ax.set_xlabel("$\\mu$")
        ax.set_ylabel("$\\delta$")
        ax.legend(loc=0)
        ax.grid(visible=True)
        sns.despine(left=True, bottom=True)
        plt.show(fig)
