# Interpreting results

## Fundamentals and notation

The measured scaling of the time-series process is $\delta$.

The $\mu$ is a complexity index, defined as the power for the inter-event time distribution, $\frac{1}{\tau^\mu}$, where $\tau$ is inter-event time.

## Frames of reference

For a totally random process, DEA yields $\delta = 0.5$.

The closer $\delta$ is to 1, and the closer $\mu$ is to 2, the more complex the data-series is. Those are the critical values of $\delta$ and $\mu$.

If $\delta < 0.5$ this may mean the time-series is not complex in the way DEA is designed to quantify (work in this area is ongoing).

## Calculating mu

Three rules of calculating $\mu$ are supported:

1. $\mu = 1 + \delta$, holds when $1 < \mu < 2$
2. $\mu = 1 + \frac{1}{\delta}$, holds when $2 < \mu < 3$
3. $\mu = 2 - \delta$, holds when $1 < \mu < 2$

The correct rule for any particular analysis may vary. Because rigorous rules for determining which is correct in what situation have yet to be laid down, all candidates are calculated and plotted so that users can compare them. Typically the correct value for will lie along the line, since that line represents the theoretical relationship. If you already have an expectation for what rule $\mu$ should follow, e.g., from theoretical arguments, use that.

The theoretical justifications for the two methods of deriving from the scaling are given in Section 3.2 of Scafetta, N., & Grigolini, P. ([2002](https://doi.org/10.1103/PhysRevE.66.036130)) and Section 3.1‒3.2 of Grigolini, P., Palatella, L., & Raffaelli, G. ([2001](https://doi.org/10.1142/S0218348X01000865)).
