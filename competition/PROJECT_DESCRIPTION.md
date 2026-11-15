# AURORA-SR: Earth-observation research

**NASA Space Apps 2026 challenge fit:** Be An Earth System Trend Detective!  
**Team:** ASTRO BADDIES

## What it does and how it works

AURORA-SR is primarily a research and model-development project. We develop and evaluate Earth-observation models for spectral reconstruction and drought analysis, with the aim of examining how climate conditions and land-surface signals can help explain environmental change. The website is an access layer for selected model outputs, figures, and a focused trend-analysis demonstration; it is not the research or model-training pipeline.

NASA POWER is the primary climate dataset in our drought research. The workflow requests daily Agroclimatology (AG) records for corrected precipitation (`PRECTOTCORR`) and 2-metre mean, maximum, and minimum air temperature (`T2M`, `T2M_MAX`, `T2M_MIN`). It derives 7-, 15-, 30-, 60-, and 90-day climate features and aligns them with Sentinel-2 scene observations. Rainfall over the following 30 days is reserved to construct an exploratory rainfall-deficit drought target. NASA POWER's gridded meteorological history, based on NASA GMAO MERRA-2 and GEOS-IT products, supplies the main weather context.

Sentinel-2 adds localized land-surface information through vegetation and moisture statistics. In a separate model path, SpectralNetV10 is trained to reconstruct six Sentinel-2-style bands (B02, B03, B04, B08, B11, B12) from RGB imagery. These research models and their training/evaluation are distinct from the static website.

The Trend Lab demonstrates the challenge workflow with a bundled NASA POWER monthly-derived annual series (1981-2025) for four selected Indian grid points. It offers six climate variables, region comparison, selectable years, linear change rates per decade, Newey-West HAC uncertainty, p-values, 95% confidence intervals, and CSV export. Its statistical test assesses each point's trend against a zero slope; it does not test whether two regional slopes differ. The site does not query POWER or Sentinel-2 live and does not train models in the browser.

## Benefits and intended impact

The project brings climate history, satellite surface signals, spectral reconstruction, and drought-model research into one accessible experience. The Trend Lab helps users inspect what is changing, where, by how much, and how strong the statistical evidence is. The intended impact is to make Earth-science model results easier to explore and to support further research into regional environmental change and drought, not to issue operational alerts.

## Tools and technologies

Research and modeling use Python, pandas, NumPy, scikit-learn, and PyTorch. The website uses HTML, CSS, and JavaScript and is deployed as a static Netlify site. Climate access uses the NASA POWER Daily and Monthly Point APIs. Optional Sentinel-2 scene-statistics access uses Microsoft Planetary Computer. The site presents precomputed data and research outputs; it does not require specialized spacecraft hardware.

## Creative approach and factors considered

The project combines a learned RGB-to-spectral model with physics-derived vegetation/moisture features, climate history from NASA POWER, and a regional trend interface. We considered the different spatial and temporal resolutions of POWER and Sentinel-2, missing or cloud-affected scenes, data-to-scene timing, autocorrelation, confidence and uncertainty, and proxy-label limits. NASA POWER provides broad, consistent climate context; Sentinel-2 provides the complementary surface view.

## Scientific limits

The rainfall-deficit target is a proxy, not field-measured drought, crop damage, or soil moisture. POWER grid cells are not local weather stations. The Trend Lab is limited to its four bundled points and annual series; significance does not establish cause or impact. Model confidence is not calibrated and the website's charts and figures are selected, precomputed examples. These limitations are documented so results are not mistaken for operational forecasts or NASA endorsement.

**Data sources:** [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), [NASA POWER Monthly API](https://power.larc.nasa.gov/docs/services/api/temporal/monthly/), and [NASA POWER data sources and methodology](https://power.larc.nasa.gov/docs/faqs/data/).
