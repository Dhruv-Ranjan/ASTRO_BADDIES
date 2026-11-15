# ASTRO BADDIES — Earth Change Explorer

Earth's environmental systems are always changing. A variable can rise in one
region and fall in another, even when the same process connects both places.
ASTRO BADDIES is a research prototype shaped around the challenge of finding
those changes: choose a variable measured by an Earth science mission or
produced by a model, view how it changes over time and space, estimate the
size and direction of change, and judge whether the evidence supports a
statistically significant trend.

## Product direction

The intended experience starts with a science variable rather than a fixed
alert. A researcher should be able to identify a variable and its source,
choose a time period and region, compare local directions of change, and
inspect trend magnitude, uncertainty, and significance with enough context to
interpret the result. The interface should distinguish observations,
reconstructed values, model estimates, and simulations.

## What this release demonstrates

The current release is a drought-focused research demonstrator and is not yet
the general-purpose trend-analysis workflow described above. Its lead
experience is the Spectral Lab, which displays six Sentinel-2-style band
previews reconstructed from RGB imagery by `SpectralNetV10`. A bundled
regional dashboard shows experimental drought outputs and lets visitors
replay a resource-aware observation planner.

The hosted Spectral Lab displays a recorded, verified result; it does not run
inference in the browser. Arbitrary image inference requires the local Python
API and checkpoint. The drought labels are predominantly proxy-derived, risk
scores are not calibrated probabilities, forecast extrapolation is
experimental, and planner constraints and belief updates are simulated.

## Pipeline in this prototype

1. A local RGB image is passed to the spectral model.
2. `SpectralNetV10` reconstructs B02, B03, B04, B08, B11, and B12.
3. Temporal features and spectral indices feed the drought estimation code.
4. The planner ranks candidate observations by information gain per resource
   cost, subject to simulated battery and storage constraints.

This pipeline is a foundation for broader Earth-variable exploration, not a
claim that regional trend significance analysis is already implemented.
