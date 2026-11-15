# Architecture

## Pipeline

1. The frontend accepts an RGB image or the repository sample.
2. The local API normalizes the image and invokes `SpectralNetV10`.
3. The model emits six reconstructed channels: B02, B03, B04, B08, B11, and
   B12.
4. Temporal drought code consumes spectral/sequence features and produces
   model-derived drought risk.
5. `AURORA/integrated_mission.py` converts forecast rows into candidates.
6. Candidates are scored by information gain per energy cost, subject to
   battery and storage constraints.
7. `update_belief_after_observation` applies a transparent simulated posterior
   update; it does not create observed satellite data.

## NASA POWER climate workflow

`AURORA/AURORA_REALWORLD_FINAL.py` is a separate research path that pairs
Sentinel-2 scene statistics with NASA POWER daily weather. It requests
corrected precipitation (`PRECTOTCORR`) and mean, maximum, and minimum
temperature at the study-area centroid through NASA POWER's Daily Point API.
Rolling climate windows and a prior-year rainfall anomaly form the predictors;
the next 30 days of rainfall define a drought-proxy target and are excluded
from predictor features. A backward timestamp merge aligns the two data
sources. Since the rolling windows include the matched calendar-day POWER
record, this implementation does not guarantee intraday-causal predictors;
lag the daily features before asserting real-time forecast validity. NASA
POWER provides the main climate context, while Sentinel-2 supplies
complementary surface observations. This research workflow is distinct from
the hosted dashboard's bundled outputs.

## Main code

- `backend/mission_api.py`: local HTTP inference service.
- `FINAL_PROJECT_NASA/src/model_defs.py`: spectral model definitions.
- `FINAL_PROJECT_NASA/src/realworld_drought_production.py`: drought pipeline.
- `AURORA/AURORA_REALWORLD_FINAL.py`: NASA POWER + Sentinel-2 research pipeline.
- `AURORA/integrated_mission.py`: mission selection and belief update.
- `frontend/mission-control.html`: browser dashboard.

## Deployment boundary

Netlify serves the static frontend only. The Python API is local/demo
infrastructure and is not deployed by `frontend/netlify.toml`.

