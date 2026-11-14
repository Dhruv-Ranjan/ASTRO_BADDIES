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

## Main code

- `backend/mission_api.py`: local HTTP inference service.
- `FINAL_PROJECT_NASA/src/model_defs.py`: spectral model definitions.
- `FINAL_PROJECT_NASA/src/realworld_drought_production.py`: drought pipeline.
- `AURORA/integrated_mission.py`: mission selection and belief update.
- `frontend/mission-control.html`: browser dashboard.

## Deployment boundary

Netlify serves the static frontend only. The Python API is local/demo
infrastructure and is not deployed by `frontend/netlify.toml`.

