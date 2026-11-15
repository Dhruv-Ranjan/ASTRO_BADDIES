# Technical overview

The system combines a `SpectralNetV10` RGB-to-six-band model, temporal drought
forecasting code, and the AURORA resource-aware planner. The planner consumes
forecast rows, ranks candidate regions, selects a feasible target, and records
a simulated belief update. The frontend is static and the local Python API
provides optional checkpoint-backed inference.

The principal climate-data path is the distinct research workflow
`AURORA/AURORA_REALWORLD_FINAL.py`. NASA POWER Daily Point API data
(`PRECTOTCORR`, `T2M`, `T2M_MAX`, `T2M_MIN`) provide daily precipitation and
temperature context at the study-area centroid. The code derives rolling
rolling features, aligns these to Sentinel-2 scene statistics, and predicts a
future 30-day rainfall-deficit proxy target. The rolling windows include the
matched calendar day, so apply a daily lag before claiming intraday-causal
forecasting. Sentinel-2 is the complementary
surface-observation source. NASA POWER's NASA GMAO MERRA-2 / GEOS-IT-based
meteorology is the backbone of this combined climate analysis; neither source
is fetched by the static hosted website.

