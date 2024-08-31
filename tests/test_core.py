"""Tests for the core methods."""

from typing import Self

import numpy as np

from src.pymdea.core import DeaEngine, DeaLoader


class TestLoader:
    """Tests for the DeaLoader."""

    def test_make_sample_data(self: Self) -> None:
        """Test the make_sample_data method."""
        dea_loader = DeaLoader()
        dea_loader.make_sample_data(1000)
        assert hasattr(dea_loader, "seed"), "Seed is missing."
        assert hasattr(dea_loader, "data"), "Data is missing."
        assert isinstance(dea_loader.seed, int), "Seed is not an int."
        assert isinstance(dea_loader.data, np.ndarray), "Data is not a numpy ndarray."
        assert np.all(np.isfinite(dea_loader.data)), "Data contains non-finite values."

    def test_make_diffusion_process(self: Self) -> Self:
        """Test the make_diffusion_process method."""
        dea_loader = DeaLoader()
        dea_loader.make_diffusion_process()
        assert hasattr(dea_loader, "data"), "Data is missing."
        assert isinstance(dea_loader.data, np.ndarray), "Data is not a numpy ndarray."
        assert np.all(np.isfinite(dea_loader.data)), "Data contains non-finite values."


class TestEngine:
    """Tests for the DeaEngine."""

    def test_apply_stripes(self: Self) -> None:
        """Test the _apply_stripes method."""
        dea_loader = DeaLoader()
        dea_loader.make_sample_data(1000)
        dea_engine = DeaEngine(dea_loader)
        dea_engine.analyze_with_stripes(fit_start=10, fit_stop=100)
        assert hasattr(dea_engine, "series"), "Series is missing."
        assert isinstance(
            dea_engine.series,
            np.ndarray,
        ), "Series is not a numpy ndarray."
        assert np.all(
            np.isfinite(dea_engine.series),
        ), "Series contains non-finite values."
        assert len(dea_engine.data) == len(
            dea_engine.series,
        ), "Series does not have the same length as the data."

    def test_get_events(self: Self) -> None:
        """Test the _get_events method."""
        dea_loader = DeaLoader()
        dea_loader.make_sample_data(1000)
        dea_engine = DeaEngine(dea_loader)
        dea_engine.analyze_with_stripes(fit_start=10, fit_stop=100)
        assert hasattr(dea_engine, "events"), "Events is missing."
        assert isinstance(
            dea_engine.events,
            np.ndarray,
        ), "Events is not a numpy ndarray."
        assert np.isfinite(
            dea_engine.events,
        ).all(), "Events contains non-finite values."
        assert np.isin(
            dea_engine.events,
            [0, 1],
        ).all(), "Events contains values other than 0 or 1."
        assert len(dea_engine.series) == len(
            dea_engine.events,
        ), "Events does not have the same length as the series."

    def test_make_trajectory(self: Self) -> None:
        """Test the _make_trajectory method."""
        dea_loader = DeaLoader()
        dea_loader.make_sample_data(1000)
        dea_engine = DeaEngine(dea_loader)
        dea_engine.analyze_with_stripes(fit_start=10, fit_stop=100)
        assert hasattr(dea_engine, "trajectory"), "Trajectory is missing."
        assert isinstance(
            dea_engine.trajectory,
            np.ndarray,
        ), "Trajectory is not a numpy ndarray."
        assert np.isfinite(
            dea_engine.trajectory,
        ).all(), "Trajectory contains non-finite values."
        assert len(dea_engine.events) == len(
            dea_engine.trajectory,
        ), "Trajectory does not have the same length as the events."

    def test_calculate_entropy(self: Self) -> None:
        """Test the _calculate_entropy method."""
        dea_loader = DeaLoader()
        dea_loader.make_sample_data(1000)
        dea_engine = DeaEngine(dea_loader)
        dea_engine.analyze_with_stripes(fit_start=10, fit_stop=100)
        assert hasattr(dea_engine, "window_lengths"), "Window lengths are missing."
        assert hasattr(dea_engine, "entropies"), "Entropies are missing."
        assert isinstance(
            dea_engine.window_lengths,
            np.ndarray,
        ), "Window lengths is not a numpy ndarray."
        assert isinstance(
            dea_engine.entropies,
            np.ndarray,
        ), "Entropies is not a numpy ndarray."
        assert np.isfinite(
            dea_engine.window_lengths,
        ).all(), "Window lengths contains non-finite values."
        assert np.isfinite(
            dea_engine.entropies,
        ).all(), "Entropies contains non-finite values."
        assert len(dea_engine.window_lengths) == len(
            dea_engine.entropies,
        ), "Entropies does not have the same length as window lengths."

    def test_calculate_scaling(self: Self) -> None:
        """Test the _calculate_scaling method."""
        dea_loader = DeaLoader()
        dea_loader.make_sample_data(1000)
        dea_engine = DeaEngine(dea_loader)
        dea_engine.analyze_with_stripes(fit_start=10, fit_stop=100)
        assert hasattr(dea_engine, "length_slice"), "Length slice is missing."
        assert hasattr(dea_engine, "fit_coefficients"), "Fit coefficients are missing."
        assert hasattr(dea_engine, "delta"), "Delta is missing."
        assert isinstance(dea_engine.delta, float), "Delta is not a float."
        assert isinstance(
            dea_engine.length_slice,
            np.ndarray,
        ), "Length slice is not a numpy ndarray."
        assert (
            len(dea_engine.fit_coefficients) == 2  # noqa: PLR2004
        ), "Fit coefficients does not have length 2."
        assert np.isfinite(
            dea_engine.fit_coefficients,
        ).all(), "Fit coefficients contains non-finite values."
        assert np.isfinite(dea_engine.delta), "Delta is non-finite."
        assert dea_engine.delta > 0, "Delta is negative."

    def test_calculate_mu(self: Self) -> None:
        """Test the _calculate_mu method."""
        dea_loader = DeaLoader()
        dea_loader.make_sample_data(1000)
        dea_engine = DeaEngine(dea_loader)
        dea_engine.analyze_with_stripes(fit_start=10, fit_stop=100)
        assert hasattr(dea_engine, "mu1"), "mu1 is missing"
        assert hasattr(dea_engine, "mu2"), "mu2 is missing"
        assert isinstance(dea_engine.mu1, float), "mu1 is not a float."
        assert isinstance(dea_engine.mu2, float), "mu2 is not a float."
        assert np.isfinite(dea_engine.mu1), "mu1 is non-finite."
        assert np.isfinite(dea_engine.mu1), "mu1 is non-finite."
        assert dea_engine.mu1 > 0, "mu1 is negative."
        assert dea_engine.mu2 > 0, "mu2 is negative."


if __name__ == "__main__":
    test_loader = TestLoader()
    test_loader.test_make_sample_data()

    test_engine = TestEngine()
    test_engine.test_apply_stripes()
    test_engine.test_get_events()
    test_engine.test_make_trajectory()
    test_engine.test_get_entropy()
    test_engine.test_get_scaling()
