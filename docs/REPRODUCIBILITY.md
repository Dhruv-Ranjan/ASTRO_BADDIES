# Reproducibility

## Inference

Install `FINAL_PROJECT_NASA/requirements.txt` and
`backend/requirements-research.txt`, then run:

```powershell
.\.venv\Scripts\python.exe .\backend\mission_api.py
```

The release includes the selected production checkpoints below 100 MB per
file. The large sequence tensor dataset is not included.

## NASA POWER drought workflow

The distinct NASA POWER + Sentinel-2 analysis is run with
`AURORA/AURORA_REALWORLD_FINAL.py`. Provide a Sentinel-2 scene-statistics CSV
with timestamps and the expected spectral summaries, then run:

```powershell
python .\AURORA\AURORA_REALWORLD_FINAL.py --input .\path\to\sentinel2_scenes.csv --output-dir .\AURORA_REALWORLD_OUTPUT
```

With network access, the script calls NASA POWER's Daily Point API for
`PRECTOTCORR`, `T2M`, `T2M_MAX`, and `T2M_MIN` at the study-area centroid and
caches the result as `nasa_power_daily.csv`. It builds preceding 7/15/30/60/90
day climate summaries and aligns them to Sentinel timestamps; the subsequent
30-day rainfall-deficit target is reserved for evaluation. The current rolling
features include the matched calendar-day POWER record, so lag daily inputs
before claiming intraday-causal forecasts. POWER climate data are not bundled,
and this pipeline's target is a rainfall-derived proxy, not field-verified
drought impact. See [`DATA_SOURCES.md`](DATA_SOURCES.md).

## Training and evaluation

The complete pipeline is launched with:

```powershell
.\.venv\Scripts\python.exe .\FINAL_PROJECT_NASA\run_project.py
```

Restore the exact tensor dataset and any manifest described by
`FINAL_PROJECT_NASA/README.md` before expecting production metrics. Run the
focused tests after restoration.

## Simulation

Use `AURORA/integrated_mission.py` with forecast rows and call
`plan_mission(...)`, followed by `update_belief_after_observation(...)`.
The planner is deterministic for fixed rows/resources except for its
timestamp field.

## Asset policy

Excluded assets are listed in `competition/SUBMISSION_CHECKLIST.md`. They are
not replaced with fake weights. Checksums should be recorded by whoever
downloads the private/large inputs for a formal reproduction run.

