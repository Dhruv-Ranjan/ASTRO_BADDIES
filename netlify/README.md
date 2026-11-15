# AURORA-SR Netlify site

This folder deploys AURORA-SR's public research interface. The home page is the
Trend Lab: it compares annual NASA POWER values across selected grid points and
calculates trend magnitudes and uncertainty from its bundled dataset. Research
and model training happen in the project's Python workflows; the website
exposes selected results and a focused climate-trend demonstration.

## Deploy

Upload this folder as the site directory, or connect the repository and set:

```text
Base directory: ASTRO_BADDIES/netlify
Build command: leave blank
Publish directory: .
```

## Pages

- `index.html` — NASA POWER Trend Lab home page.
- `data/power-annual-1981-2025.json` — annual series derived from NASA POWER's
  monthly Agroclimatology point API for four selected Indian grid points.
- `spectral-lab.html` — precomputed six-band reconstruction interface.
- `mission-control.html` — research console with bundled drought regions,
  planner replay, scenarios, and research notes.
- `graphs.html` — browsable recorded experiment figures.

The static demo displays a precomputed reconstruction. It does not include
private PyTorch checkpoints or a hosted inference API. Arbitrary-image
inference is available through the local API described in
`../docs/DEMO_GUIDE.md`.

Experiment figures under `graphs/` are recorded research outputs, not live
spacecraft telemetry or current Earth monitoring data.

The site does not make live NASA POWER or Sentinel-2 requests. The Trend Lab
uses the checked-in annual NASA POWER snapshot and performs its selected trend
analysis in the browser. NASA POWER is also the primary daily climate input to
the separate `AURORA/AURORA_REALWORLD_FINAL.py` drought research workflow:
precipitation and temperature become rolling climate features and a future
rainfall-deficit proxy target. Sentinel-2 scene statistics add complementary
land-surface signals. See [`../docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md)
for data lineage and scientific limits.

## Local preview

From this directory:

```powershell
python -m http.server 8080
```

Open `http://127.0.0.1:8080/` for the Trend Lab,
`http://127.0.0.1:8080/spectral-lab.html` for the Spectral Lab, or
`http://127.0.0.1:8080/mission-control.html` for the Research Console.
