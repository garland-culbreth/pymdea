"""Tests for the core methods."""

from typing import Self

import matplotlib.pyplot as plt

from src.pymdea.core import DeaEngine, DeaLoader
from src.pymdea.plot import DeaPlotter


class TestPlotter:
    """Tests for the DeaEngine."""

    def test_s_vs_l(self: Self) -> None:
        """Test the _calculate_mu method."""
        dea_loader = DeaLoader()
        dea_loader.make_sample_data(1000)
        dea_engine = DeaEngine(dea_loader)
        dea_engine.analyze_with_stripes(fit_start=0.1, fit_stop=0.9)
        dea_plotter = DeaPlotter(dea_engine)
        dea_plotter.s_vs_l()
        assert hasattr(dea_plotter, "fig_s_vs_l"), "fig_s_vs_l is missing"
        assert isinstance(
            dea_plotter.fig_s_vs_l, plt.Figure,
        ), "fig_s_vs_l is not a matplotlib figure."

    def test_mu_candidates(self: Self) -> None:
        """Test the _calculate_mu method."""
        dea_loader = DeaLoader()
        dea_loader.make_sample_data(1000)
        dea_engine = DeaEngine(dea_loader)
        dea_engine.analyze_with_stripes(fit_start=0.1, fit_stop=0.9)
        dea_plotter = DeaPlotter(dea_engine)
        dea_plotter.mu_candidates()
        assert hasattr(dea_plotter, "fig_mu_candidates"), "fig_mu_candidates is missing"
        assert isinstance(
            dea_plotter.fig_mu_candidates, plt.Figure,
        ), "fig_mu_candidates is not a matplotlib figure."


if __name__ == "__main__":
    test_plotter = TestPlotter()
    test_plotter.test_s_vs_l()
    test_plotter.test_mu_candidates()
