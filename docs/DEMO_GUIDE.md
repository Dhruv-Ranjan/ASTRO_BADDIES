# Demo guide

## Hosted demo

Open the Trend Lab at https://aurora-sr.netlify.app/. Select a NASA POWER
climate variable, two study grid points, and the analysis years to compare
annual records, estimated rates of change, confidence intervals, and trend
significance. The browser analyzes an annual dataset bundled in the site; it
does not request NASA POWER live. Open Spectral Lab to inspect the recorded
six-band reconstruction. It serves a precomputed model result and does not
run new inference in the browser.

Use **Research console** to explore the bundled regional drought snapshot,
resource-planning replay, scenario controls, and research notes. The snapshot
and simulated mission choices are not live satellite telemetry.

The hosted pages do not make live NASA POWER requests or download Sentinel-2
scenes. The Trend Lab's checked-in annual sample is derived from NASA POWER's
public monthly Agroclimatology point API. NASA POWER is also used by a separate
daily drought research pipeline in
`AURORA/AURORA_REALWORLD_FINAL.py`: it supplies daily precipitation and
temperature context, which is combined with timestamped Sentinel-2
scene-statistics to build drought predictors and a future rainfall-deficit
proxy target. The hosted dashboard's bundled drought snapshot and reported
demo metrics should not be attributed to a live NASA POWER query. See
[`DATA_SOURCES.md`](DATA_SOURCES.md) for detail.

## Local static preview

From the `ASTRO_BADDIES` repository root:

```powershell
python -m http.server 8080 --directory .\netlify
```

Open `http://127.0.0.1:8080/` for the Spectral Lab or
`http://127.0.0.1:8080/mission-control.html` for the dashboard.

## Local model inference

The separate Python API supports arbitrary PNG/JPEG input when its model
checkpoint and dependencies are available:

```powershell
python .\backend\mission_api.py
```

Use the local API only for development. The hosted static site does not expose
this service.

## Explain the limits

The Trend Lab offers six NASA POWER climate variables for four bundled grid
points, not a general mission/model catalogue or full spatial analysis. Its
linear trend test uses a large-sample normal approximation and Newey-West HAC
standard errors; significance applies to each individual time series, not the
difference between two regional slopes. It does not establish causes or
impacts. Drought labels are predominantly proxy-derived, confidence outputs
are uncalibrated, and mission planning uses simulated resource decisions. Do
not describe these outputs as operational alerts or spacecraft telemetry.
