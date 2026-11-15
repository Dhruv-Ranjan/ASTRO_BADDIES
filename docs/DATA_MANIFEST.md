# Data and asset manifest

## Included assets

The two selected checkpoints are included because each is below GitHub's
per-file limit and is needed for the local demo/API:

| Path | Role |
| --- | --- |
| `FINAL_PROJECT_NASA/artifacts/spectral_realworld_production.pt` | RGB-to-six-band inference |
| `FINAL_PROJECT_NASA/artifacts/drought_realworld_production_model.pt` | Drought model artifact for local evaluation |
| `backend/demo_assets/test_rgb_sample.png` | Small public demo input |

SHA-256:

```text
221D03D802CC142584379A81E7E528C545C4ECE143422127DF47C3579758FA68  FINAL_PROJECT_NASA/artifacts/spectral_realworld_production.pt
5D9BB4833F9B6638AC54F51C50654AE76D80A78D57DB2B412860E4E11CCD5312  FINAL_PROJECT_NASA/artifacts/drought_realworld_production_model.pt
E53F541D953AEB42091B1D80FBD95857F92DEFB3D6E4284EA2C64BF3F1E4CA79  backend/demo_assets/test_rgb_sample.png
```

Recompute them before a formal submission with:

```powershell
Get-FileHash .\FINAL_PROJECT_NASA\artifacts\*.pt -Algorithm SHA256
```

## Excluded assets

The sequence tensor dataset (`*.npz`, approximately 1.1 GB), remote manifests,
downloaded scenes, caches, databases, generated logs, bytecode, and virtual
environments are excluded. They are not silently replaced with synthetic
weights. Exact retraining requires separately acquiring the inputs described in
the reproducibility documentation.

The NASA POWER daily CSV (`nasa_power_daily.csv`) is a generated cache created
by `AURORA/AURORA_REALWORLD_FINAL.py` inside its selected output directory; it
is not committed in this release. To reproduce that workflow, the script
requests NASA POWER daily `PRECTOTCORR`, `T2M`, `T2M_MAX`, and `T2M_MIN` at the
study-area centroid, then derives rolling climate features and a
future 30-day rainfall-deficit target. See [`DATA_SOURCES.md`](DATA_SOURCES.md)
for full feature and interpretation details. Sentinel-2 scene-statistics
inputs are locally supplied or optionally downloaded; they are not bundled.

