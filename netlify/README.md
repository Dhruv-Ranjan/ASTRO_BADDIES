# ASTRO BADDIES Netlify site

This folder is the static deployment for the ASTRO BADDIES Earth Change
Explorer. The home page opens on the Spectral Lab, where visitors can inspect
the checked-in six-band reconstruction sample and its recorded reflectance
summaries.

## Deploy

Upload this folder as the site directory, or connect the repository and set:

```text
Base directory: ASTRO_BADDIES/netlify
Build command: leave blank
Publish directory: .
```

## Pages

- `index.html` — Spectral Lab home page.
- `mission-control.html` — bundled drought regions, mission planner replay,
  spectral demo, scenarios, and research notes.
- `graphs.html` — browsable experiment figures.

The static demo displays a precomputed reconstruction. It does not include
private PyTorch checkpoints or a hosted inference API. Arbitrary-image
inference is available through the local API described in
`../docs/DEMO_GUIDE.md`.

Experiment figures under `graphs/` are recorded research outputs, not live
spacecraft telemetry or current Earth monitoring data.

## Local preview

From this directory:

```powershell
python -m http.server 8080
```

Open `http://127.0.0.1:8080/` for the Spectral Lab, or
`http://127.0.0.1:8080/mission-control.html` for the dashboard.
