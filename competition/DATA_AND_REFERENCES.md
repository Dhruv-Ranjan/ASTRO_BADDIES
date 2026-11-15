# Data and references

## NASA POWER (primary climate source)

The dedicated real-world drought workflow in
[`../AURORA/AURORA_REALWORLD_FINAL.py`](../AURORA/AURORA_REALWORLD_FINAL.py)
requests daily NASA POWER Agroclimatology data at the study-area centroid. It
uses corrected precipitation (`PRECTOTCORR`) and mean, maximum, and minimum
temperature (`T2M`, `T2M_MAX`, `T2M_MIN`). The script creates rolling
rainfall/temperature features, aligns them to Sentinel-2 scene timestamps,
and defines the next 30 days' rainfall deficit as a meteorological drought
proxy target. NASA POWER is the principal climate backbone in this workflow.
Its meteorology is drawn from NASA GMAO MERRA-2 / GEOS-IT products and is
gridded climate context, not field observation.

The next 30-day target is excluded from predictors. Rolling features include
the matched calendar day's POWER row, however, so date-level summaries may
include data later than an intraday Sentinel observation. Lag the daily
features before claiming strict real-time forecast causality. See the full
timing caveat in [data provenance](../docs/DATA_SOURCES.md).

## Sentinel-2 (complementary surface data)

Sentinel-2 scene-level vegetation and moisture summaries supply local surface
signals to combine with NASA POWER's daily weather context. The Spectral Lab
also displays six Sentinel-2-style bands reconstructed from RGB. Scene
statistics, raw imagery, and the generated NASA POWER cache are not bundled;
the static website displays precomputed artifacts and does not fetch either
source live.

See [../docs/DATA_SOURCES.md](../docs/DATA_SOURCES.md) for dataset provenance,
reproduction steps, and license boundaries. Official references: [NASA POWER
Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) and
[NASA POWER data sources](https://power.larc.nasa.gov/docs/faqs/data/). NASA
data terms and image-provider terms apply to their respective data.

