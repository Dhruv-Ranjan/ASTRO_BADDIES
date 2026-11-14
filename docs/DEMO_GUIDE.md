# Demo guide

## Live demo

Open https://aurora-sr.netlify.app/ and walk through Overview, Spectral Lab,
Drought Lab, Mission Control, Scenarios, and Research.

## Local demo

```powershell
.\.venv\Scripts\python.exe .\backend\mission_api.py
```

Open `http://127.0.0.1:8000/mission-control.html`, choose **Use repository
sample**, and select **Run SpectralNetV10**. Explain that the six previews are
model outputs from an RGB sample, not newly acquired imagery. Then show the
mission-control and scenario views and point out the integrity note.

## Recording

Use the live or local dashboard, record one complete pass from RGB input to
six-band output and mission selection, and state the limitations aloud. Do not
describe simulated resource decisions as spacecraft telemetry.

