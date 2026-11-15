# Submission checklist

- [x] Project description, technical overview, demo script, and AI disclosure.
- [x] README, architecture, deployment, data, testing, and reproducibility docs.
- [x] Static frontend and local inference API source.
- [x] Selected checkpoints below GitHub's 100 MB per-file limit.
- [x] Genuine research figures and sample input.
- [ ] Capture and add the six browser-state screenshots listed in
      `docs/screenshots/README.md`.
- [ ] Confirm data-provider terms and attribution for any raw imagery added
      later.
- [ ] Restore the large tensor dataset and run a full reproduction if claiming
      full retraining reproducibility.
- [x] Document NASA POWER's role, requested variables, derived features,
      Sentinel-2 complement, and data/target limitations.

## Intentionally excluded

- `dataset_seq_forecast_production_real.npz` and related `.npz` files:
  approximately 1.1 GB+ training inputs.
- Remote caches, downloaded scene data, databases, logs, bytecode, and local
  virtual environments.
- The generated `nasa_power_daily.csv` cache; the documented pipeline obtains
  it from NASA POWER's public Daily Point API when run.
- Any private credentials or local configuration.

