# Demo guide

## Hosted demo

Open the Spectral Lab at https://aurora-sr.netlify.app/. Select B02, B03, B04,
B08, B11, or B12 to inspect the recorded reconstruction preview and its
reflectance summary. The hosted page serves a precomputed local model result;
it does not run new inference in the browser.

Use **Mission dashboard** to explore the bundled regional drought snapshot,
resource-planning replay, scenario controls, and research notes. The snapshot
and simulated mission choices are not live satellite telemetry.

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

The current release demonstrates spectral reconstruction and drought-focused
research outputs. It does not yet provide a general variable catalogue,
spatiotemporal trend fitting, or statistical significance testing. Drought
labels are predominantly proxy-derived, confidence outputs are uncalibrated,
and the mission planner uses simulated resource decisions. Do not describe
these outputs as operational alerts or spacecraft telemetry.
