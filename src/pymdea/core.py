"""Diffusion entropy analysis core methods"""

from typing import Self, Literal
import os
import logging
import numpy as np
import polars as pl
from scipy import stats
from tqdm import tqdm


class DeaLoader:
    def __init__(self) -> Self:
        pass

    def make_sample_data(self, length: int = 100000, seed: int = 1) -> np.ndarray:
        """Generates an array of sample data."""
        np.random.seed(seed)  # for baseline consistency 1010
        random_steps = np.random.choice([-1, 1], length)
        random_steps[0] = 0  # always start from 0
        random_walk = np.cumsum(random_steps)
        self.seed = seed
        self.data = random_walk
        return self

    def read_data_file(self, filepath: str, column_name: str) -> pl.DataFrame:
        """
        Convenience function for reading input data

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
        AssertionError
            If filepath points to a file of type other than
            CSV. Support for more types of files is a work in
            progress.
        """
        filetype = os.path.splitext(filepath)[1]
        supported_types = [".csv"]
        assert (
            filetype in supported_types
        ), f"'filetype' must be one of: {supported_types}."
        if filetype == ".csv":
            self.data = pl.scan_csv(filepath).select(column_name).to_numpy()
        return self


class DeaEngine:
    def __init__(self, loader: DeaLoader) -> Self:
        self.data = loader.data

    def _apply_stripes(self) -> Self:
        """Rounds `data` to `stripes` evenly spaced intervals"""
        if np.min(self.data) <= 0:
            self.data = self.data + np.abs(np.min(self.data))
        elif np.min(self.data) > 0:
            self.data = self.data - np.abs(np.min(self.data))
        max_data = np.max(self.data)
        min_data = np.min(self.data)
        data_width = np.abs(max_data - min_data)
        stripe_size = data_width / self.number_of_stripes
        self.series = self.data / stripe_size
        return self

    def _get_events(self) -> Self:
        """Records an event (1) when `series` changes value."""
        events = np.zeros(len(self.series))
        for i in range(1, len(self.series)):
            if (
                self.series[i] < np.floor(self.series[i - 1]) + 1
                and self.series[i] > np.ceil(self.series[i - 1]) - 1
            ):
                next  # if both true, no crossing
            else:
                events[i] = 1
        np.append(events, 0)
        self.events = events
        return self

    def _make_trajectory(self) -> Self:
        """Constructs diffusion trajectory from events."""
        self.trajectory = np.cumsum(self.events)
        return self

    def _get_entropy(self, window_stop: float = 0.25) -> Self:
        """
        Calculates the Shannon Entropy of the diffusion trajectory

        Parameters
        ----------
        window_stop : float, optional, default: 0.25
            Proportion of total data length at which to cap the maximum
            window length. Large window lengths rarely produce
            informative entropy.
        """
        entropies = []
        window_lengths = np.arange(1, int(window_stop * len(self.trajectory)), 1)
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
                -np.sum(distribution * np.log(distribution)) + np.log(binsize)
            )
        self.entropies = entropies
        self.window_lengths = window_lengths
        return self

    def get_scaling(
        self,
        fit_start: int,
        fit_stop: int,
        fit_method: str = "siegel",
    ) -> Self:
        """
        Calculate scaling.

        Calculates the scaling of the input time-series by performing
        a linear fit over S(l) vs ln(l).

        Parameters
        ----------
        fit_start : int
            Index at which to start the fit slice.
        fit_stop : int
            Index at which to stop the fit slice.
        fit_method : str {"siegel", "theilsen", "ls"}, optional, default: "siegel"
            Linear fit method to use.

        Returns
        ----------
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
        supported_methods = ["siegel", "theilsen", "ls"]
        assert fit_method in supported_methods, f"""'method' must be one of:
            {supported_methods}"""
        s_slice = self.entropies[fit_start:fit_stop]
        length_slice = self.window_lengths[fit_start:fit_stop]
        if fit_method == "ls":
            logging.warning(
                """Least-squares linear fits can introduce systematic error when applied to log-scale data. Prefer the more robust 'theilsen' or 'siegel' methods."""
            )
            coefficients = np.polyfit(np.log(length_slice), s_slice, 1)
        if fit_method == "theilsen":
            coefficients = stats.theilslopes(s_slice, np.log(length_slice))
        if fit_method == "siegel":
            coefficients = stats.siegelslopes(s_slice, np.log(length_slice))
        self.length_slice = length_slice
        self.fit_coefficients = coefficients
        self.delta = coefficients[0]
        return self

    def get_mu(self) -> Self:
        """
        Calculate powerlaw index for inter-event time distribution.

        - mu1 is the index calculated by the rule `1 + delta`.
        - mu2 is the index calculated by the rule `1 + (1 / delta)`.

        Returns
        ----------
        Self @ Engine
            Object containing the results and inputs of the diffusion
            entropy analysis.

        Notes
        ----------
        mu is calculated by both rules. later both are plotted
        against the line relating delta and mu, to hopefully
        let users graphically determine the correct mu.
        """
        self.mu1 = 1 + self.delta
        self.mu2 = 1 + (1 / self.delta)
        return self

    def _print_results(self) -> Self:
        """Print out results of analysis"""
        print("---------------------------------")
        print("result")
        print(f" δ: {self.delta}")
        print(f" μ (rule 1): {self.mu1}")
        print(f" μ (rule 2): {self.mu2}")
        print("---------------------------------")
        return self

    def analyze(
        self,
        fit_start: int,
        fit_stop: int,
        fit_method: Literal["siegel", "theilsen", "ls"] = "siegel",
        use_stripes: bool = True,
        n_stripes: int = 20
    ) -> Self:
        """
        Runs a diffusion entropy analysis.

        Parameters
        ----------
        fit_start : int
            Array index at which to start linear fit.
        fit_stop : int
            Array index at which to stop linear fit.
        fit_method : str {"siegel", "theilsen", "ls"}, optional
            Linear fit method to use. By default "siegel"

        Returns
        ----------
        Self @ Engine
            Object containing the results and inputs of the diffusion
            entropy analysis.
        """
        if use_stripes:
            assert isinstance(n_stripes, int), "n_stripes must be an integer"
            assert n_stripes > 1, "n_stripes must be greater than 1"
        self.use_stripes = use_stripes
        self.number_of_stripes = n_stripes
        if self.use_stripes:
            self._apply_stripes()
            self._get_events()
            self._make_trajectory()
        else:
            self.trajectory = self.data
        self._get_entropy()
        self.get_scaling(fit_start, fit_stop, fit_method)
        self.get_mu()
        self._print_results()
