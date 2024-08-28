# Tuning an analysis

## Choosing number of stripes

!!! warning

    Rigorous rules for choosing the proper number of stripes are still being developed.

Run a few initial tests with different values. e.g. 10, 50, 100. If there is significant disagreement in the scalings measured, vary about those trial numbers. The correct number of stripes to use is that number such that when varied up or down a little, the scaling does not change.

## Choosing fit interval

In the result figure, if the results are good, there will be a region in the loglog figure that appears linear. You want the fit interval to line up with this region. $S(l)$ is logged in calculation, $\ln(l)$ is logged by the scale of the plot. The fitting function accounts for this.
