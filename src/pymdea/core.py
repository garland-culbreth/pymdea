"""Diffusion entropy analysis core methods."""

import logging
from pathlib import Path
from typing import Literal, Self

import numpy as np
import polars as pl
import stochastic.processes.continuous
import stochastic.processes.noise
from scipy import stats
from scipy.optimize import curve_fit
from tqdm import tqdm


def _power_log(x: float, a: float, b: float) -> float:
    """Log power law for curve fit."""
    return (a * np.log(x)) + b


class DeaLoader:
    """Load data for a diffusion entropy analysis."""

    def __init__(self: Self) -> Self:
        """Load data for a diffusion entropy analysis."""

    def make_sample_data(self: Self, length: int = 100000, seed: int = 1) -> np.ndarray:
        """Generate an array of sample data.

        Parameters
        ----------
        length : int, optional, default: 100000
            Number of time-steps to produce in the sample data.
        seed : int, optional, default: 1
            Seed for random number generation.

        Returns
        -------
        Self @ Loader
            An instance of the Loader object.

        """
        rng = np.random.default_rng(seed=seed)
        random_walk = rng.choice([-1, 1], size=length).cumsum()
        random_walk[0] = 0  # always start from 0
        self.seed = seed
        self.data = random_walk
        return self

    def make_diffusion_process(
        self: Self,
        kind: Literal["cn", "gn", "fgn", "fbm"] = "gn",
        length: int = 10000,
        a: float = 0,
    ) -> Self:
        """Generate diffusion process data.

        Parameters
        ----------
        kind : str {"cn", "gn", "fgn", "fbm"}, optional, default "cn"
            Type of diffusion noise to generate. If "cn", generate a
            colored noise with spectral power `a`. If "gn", generate a
            Gaussian noise. If "fgn", generate a fractional Gaussian
            noise with Hurst index H = `a`. If "fbm", generate a
            fractional Brownian motion with Hurst index H=`a`.
        length : int, optional, default 10000
            Length of time-series to generate.
        a : float, optiona, default 0
            Only used if `kind` is "fgn", "fbm", or "cn". If `kind` is
            "fgn" or "fbm", this sets the Hurst index of the process.
            If `kind` is "cn" this sets the index of the power law
            spectrum for the noise, 1/(f^`a`).

        Returns
        -------
        Self @ Loader
            An instance of the Loader object.

        """
        if kind == "gn":
            process = stochastic.processes.noise.GaussianNoise()
            self.data = process.sample(length)
        if kind == "cn":
            process = stochastic.processes.noise.ColoredNoise(beta=a)
            self.data = process.sample(length)
        if kind == "fgn":
            process = stochastic.processes.noise.FractionalGaussianNoise(hurst=a)
            self.data = process.sample(length)
        if kind == "fbm":
            process = stochastic.processes.continuous.FractionalBrownianMotion(hurst=a)
            self.data = process.sample(length)
        return self

    def read_data_file(self: Self, filepath: str, column_name: str) -> pl.DataFrame:
        """Read input data from file.

        Parameters
        ----------
        filepath : str
            System path to a file containing data. Must include the
            full file name, including the extension. Example:
            "/example/path/to/file.csv"
        column_name : str
            Name of the column in the data file which contains the time
            series data values.

        Returns
        -------
        Self @ Loader
            An instance of the Loader object.

        Raises
        ------
        ValueError
            If filepath points to a file of type other than
            CSV. Support for more types of files is a work in
            progress.

        """
        filepath = Path(filepath)
        supported_types = [".csv"]
        if filepath.suffix not in supported_types:
            msg = f"filetype must be one of: {supported_types}."
            raise ValueError(msg)
        if filepath.suffix == ".csv":
            self.data = pl.scan_csv(filepath).select(column_name).to_numpy()
        return self


class DeaEngine:
    """Run diffusion entropy analysis."""

    def __init__(self: Self, loader: DeaLoader) -> Self:
        """Run diffusion entropy analysis."""
        self.data = loader.data

    def _apply_stripes(self: Self) -> Self:
        """Round `data` to `stripes` evenly spaced intervals."""
        if np.min(self.data) <= 0:
            self.data = self.data + np.abs(np.min(self.data))
        elif np.min(self.data) > 0:
            self.data = self.data - np.abs(np.min(self.data))
        data_width = np.abs(self.data.max() - self.data.min())
        stripe_size = data_width / self.number_of_stripes
        self.series = self.data / stripe_size
        return self

    def _get_events(self: Self) -> Self:
        """Record an event (1) when `series` changes value."""
        no_upper_crossing = self.series[1:] < np.floor(self.series[:-1]) + 1
        no_lower_crossing = self.series[1:] > np.ceil(self.series[:-1]) - 1
        events = np.where(no_lower_crossing & no_upper_crossing, 0, 1)
        self.events = np.append([0], events)  # Event impossible at index 0
        return self

    def _make_trajectory(self: Self) -> Self:
        """Construct diffusion trajectory from events."""
        self.trajectory = np.cumsum(self.events)
        return self

    def _calculate_entropy(self: Self, window_stop: float = 0.25) -> Self:
        """Calculate the Shannon Entropy of the diffusion trajectory.

        Parameters
        ----------
        window_stop : float, optional, default: 0.25
            Proportion of total data length at which to cap the maximum
            window length. Large window lengths rarely produce
            informative entropy.

        """
        entropies = []
        window_lengths = np.unique(
            np.logspace(
                start=0,
                stop=np.log10(window_stop * len(self.trajectory)),
                num=1000,
                dtype=np.int32,
            ),
        )
        for window_length in tqdm(window_lengths):
            window_starts = np.arange(0, len(self.trajectory) - window_length, 1)
            window_ends = np.arange(window_length, len(self.trajectory), 1)
            displacements = (
                self.trajectory[window_ends] - self.trajectory[window_starts]
            )
            counts, bin_edge = np.histogram(displacements, bins="doane")
            counts = np.array(counts[counts != 0])
            binsize = bin_edge[1] - bin_edge[0]
            distribution = counts / np.sum(counts)
            entropies.append(
                -np.sum(distribution * np.log(distribution)) + np.log(binsize),
            )
        self.entropies = np.asarray(entropies)
        self.window_lengths = window_lengths
        return self

    def _calculate_scaling(self: Self) -> Self:
        """Calculate scaling."""
        start_index = np.floor(self.fit_start * len(self.window_lengths)).astype(int)
        stop_index = np.floor(self.fit_stop * len(self.window_lengths)).astype(int)
        s_slice = self.entropies[start_index:stop_index]
        length_slice = self.window_lengths[start_index:stop_index]
        if self.fit_method == "ls":
            logging.warning(
                """Least-squares linear fits can introduce systematic error when
                applied to log-scale data. Prefer the more robust 'theilsen' or
                'siegel' methods.""",
            )
            coefficients = curve_fit(
                f=_power_log,
                xdata=length_slice,
                ydata=s_slice,
            )[0]  # 0 is coeffs, 1 is uncertainty, uncertainty not yet used
        if self.fit_method == "theilsen":
            coefficients = stats.theilslopes(s_slice, np.log(length_slice))
        if self.fit_method == "siegel":
            coefficients = stats.siegelslopes(s_slice, np.log(length_slice))
        self.length_slice = length_slice
        self.fit_coefficients = coefficients
        self.delta = coefficients[0]
        return self

    def _calculate_mu(self: Self) -> Self:
        """Calculate powerlaw index for inter-event time distribution.

        - mu1 is the index calculated by the rule `1 + delta`.
        - mu2 is the index calculated by the rule `1 + (1 / delta)`.

        Returns
        -------
        Self @ Engine
            Object containing the results and inputs of the diffusion
            entropy analysis.

        Notes
        -----
        mu is calculated by both rules. later both are plotted
        against the line relating delta and mu, to hopefully
        let users graphically determine the correct mu.

        """
        self.mu1 = 1 + self.delta
        self.mu2 = 1 + (1 / self.delta)
        return self

    def print_result(self: Self) -> str:
        """Print out result of analysis."""
        self.result = "--------------------------------- \n"
        self.result = self.result + "result \n"
        self.result = self.result + f" δ: {self.delta} \n"
        self.result = self.result + f" μ (rule 1): {self.mu1} \n"
        self.result = self.result + f" μ (rule 2): {self.mu2} \n"
        self.result = self.result + "---------------------------------"
        print(self.result)  # noqa: T201
        return self

    def analyze_with_stripes(
        self: Self,
        fit_start: int,
        fit_stop: int,
        fit_method: Literal["siegel", "theilsen", "ls"] = "siegel",
        n_stripes: int = 20,
    ) -> Self:
        """Run a modified diffusion entropy analysis.

        Parameters
        ----------
        fit_start : float
            Fraction of maximum window length at which to start linear fit.
        fit_stop : float
            Fraction of maximum window length at which to stop linear fit.
        fit_method : str {"siegel", "theilsen", "ls"}, optional
            Linear fit method to use. By default "siegel"
        n_stripes : int, optional, default: 20
            Number of stripes to apply to input time-series during
            analysis.

        Returns
        -------
        Self @ Engine
            Object containing the results and inputs of the diffusion
            entropy analysis.

        Raises
        ------
        ValueError
            If n_stripes < 2. At least two stripes must be applied for
            DEA to provide a meaningful result.

        Notes
        -----
        Prefer the siegel or theilsen methods. Least squares linear
        fits can introduce bias when done over log-scale data, see
        Clauset, A., Shalizi, C.R. and Newman, M.E., 2009. Power-law
        distributions in empirical data. SIAM review, 51(4), pp.661-703.
        https://doi.org/10.1137/070710111.
        https://arxiv.org/pdf/0706.1062.pdf.

        """
        if n_stripes < 2:  # noqa: PLR2004
            msg = "n_stripes must be greater than 1"
            raise ValueError(msg)
        self.number_of_stripes = n_stripes
        self.fit_start = fit_start
        self.fit_stop = fit_stop
        self.fit_method = fit_method
        self._apply_stripes()
        self._get_events()
        self._make_trajectory()
        self._calculate_entropy()
        self._calculate_scaling()
        self._calculate_mu()
        self.print_result()

    def analyze_without_stripes(
        self: Self,
        fit_start: int,
        fit_stop: int,
        fit_method: Literal["siegel", "theilsen", "ls"] = "siegel",
    ) -> Self:
        """Run a regular diffusion entropy analysis.

        Parameters
        ----------
        fit_start : float
            Fraction of maximum window length at which to start linear fit.
        fit_stop : float
            Fraction of maximum window length at which to stop linear fit.
        fit_method : str {"siegel", "theilsen", "ls"}, optional
            Linear fit method to use. By default "siegel"

        Returns
        -------
        Self @ Engine
            Object containing the results and inputs of the diffusion
            entropy analysis.

        Notes
        -----
        Prefer the siegel or theilsen methods. Least squares linear
        fits can introduce bias when done over log-scale data, see
        Clauset, A., Shalizi, C.R. and Newman, M.E., 2009. Power-law
        distributions in empirical data. SIAM review, 51(4), pp.661-703.
        https://doi.org/10.1137/070710111.
        https://arxiv.org/pdf/0706.1062.pdf.

        """
        self.trajectory = self.data
        self.fit_start = fit_start
        self.fit_stop = fit_stop
        self.fit_method = fit_method
        self._calculate_entropy()
        self._calculate_scaling()
        self._calculate_mu()
        self.print_result()
