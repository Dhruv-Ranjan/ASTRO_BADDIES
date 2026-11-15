# Data sources and licenses

## NASA POWER: primary climate context

NASA POWER (Prediction Of Worldwide Energy Resources) is the primary
meteorological source used by the dedicated real-world drought forecasting
workflow in [`AURORA/AURORA_REALWORLD_FINAL.py`](../AURORA/AURORA_REALWORLD_FINAL.py).
The script calls the [NASA POWER Daily Point API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
for the selected study area's bounding-box centroid, using the Agroclimatology
community (`AG`). It requests `PRECTOTCORR` (corrected precipitation), `T2M`
(mean temperature), `T2M_MAX`, and `T2M_MIN` over the configured date range.
NASA describes the daily service as analysis-ready time series; POWER's
meteorological data are sourced from NASA GMAO MERRA-2 and GEOS-IT
assimilation products ([source details](https://power.larc.nasa.gov/docs/faqs/data/)).

The pipeline caches the response as `nasa_power_daily.csv` in its output
directory. It derives rolling 7/15/30/60/90-day rainfall totals and averages,
temperature means, hot-day counts, and an anomaly against the prior available
365-day rainfall climatology. A backward as-of merge aligns climate conditions
to Sentinel-2 scene timestamps with a three-day maximum gap. Rolling POWER
climate features enter the predictor set, while rainfall during the subsequent
30 days is reserved for a rainfall-deficit drought proxy target. The future
target is excluded from predictors. However, the rolling features include the
matched calendar day's POWER record; as daily records can summarize a full day,
this merge alone does not guarantee intraday-causal forecasting. A one-day lag
should be applied before making a real-time forecasting claim. The target is
POWER-derived and is not independent field evidence of agricultural impacts.

NASA POWER is the climate backbone; Sentinel-2 is the complementary surface
observation. The pipeline combines its scene-level vegetation/moisture
statistics with the coarser daily weather context from POWER. Optional
Sentinel-2 L2A scene-statistics ingestion uses Microsoft Planetary Computer
catalog access. The Spectral Lab's six bands are Sentinel-2-style and its
hosted output is a bundled reconstruction. The public site does not query
either service live. POWER's gridded weather context is not point
weather-station measurements or field truth.

## Release contents and terms

The public release contains source, compact metadata, selected model artifacts,
and a small sample image. It does not contain the approximately 1.1 GB
sequence tensor dataset, downloaded scenes, or generated NASA POWER cache.
Re-running the POWER workflow requires network access to NASA's public API and
a Sentinel scene-statistics CSV (or explicitly requested optional scene
download). NASA datasets retain NASA's applicable data terms and attribution;
image-provider terms apply to Sentinel-2 distribution. Source code is MIT;
third-party data retain their own terms.

The reported SpectralNet and bundled drought-demo metrics belong to their
recorded experiments. They are not evidence that those particular metrics
were produced by the NASA POWER pipeline. For experiment-specific provenance,
see [`competition/DATA_AND_REFERENCES.md`](../competition/DATA_AND_REFERENCES.md),
the model caveats, and the AURORA output metadata.
