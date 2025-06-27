# Tuning an analysis

## Choosing number of stripes

!!! warning

    Rigorous rules for choosing the proper number of stripes are still being developed.

Run a few initial tests with different values. e.g. 10, 50, 100. If there is significant disagreement in the scalings measured, vary about those trial numbers. The correct number of stripes to use is that number such that when varied up or down a little, the scaling does not change.

## Choosing fit interval

In the result figure, if the results are good, there will be a region in the loglog figure that appears linear. You want the fit interval to line up with this region. $S(l)$ is logged in calculation, $\ln(l)$ is logged by the scale of the plot. The fitting function accounts for this.

## Choosing histogram binning method

At each window length in the analysis, diffusion entropy analysis constructs a histogram to represent the probability density function of the diffusion trajectory slices corresponding to that window length. To give valid results, the histogram must accurately represent the probability density function.

There is no histogram method that's always best. You can read more about the available methods in NumPy's [`numpy.histogram_bin_edges`](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges) documentation. The default method for pymdea is the `doane` method; it performs reasonably well in most cases and is better suited to non-normal distributions than some of the others.

## Choosing the number of window lengths

The `window_length_stop` option in `DeaEngine` specifies what the longest window length used in the analysis should be, as a proportion of the data series' length. The default setting for this options is `0.25`; that is 1/4th the overall data series length. Window lengths longer than this often begin to have difficulty resolving the underlying probability density. In some cases longer window lengths can be reasonable.

As window length approaches the overall length of the data series there are fewer and fewer diffusion trajectories, and consequently the histogram for the window length's diffusion trajectories begins to poorly represent the probability density. This is apparent when plotting $S(l)$ vs $\ln(l)$, e.g., with `DeaPlot.s_vs_l`. When the window lengths become too long to give good statistics, the corresponding entropies rapidly decreases and fall off.
