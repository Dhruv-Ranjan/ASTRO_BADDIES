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

## Intentionally excluded

- `dataset_seq_forecast_production_real.npz` and related `.npz` files:
  approximately 1.1 GB+ training inputs.
- Remote caches, downloaded scene data, databases, logs, bytecode, and local
  virtual environments.
- Any private credentials or local configuration.

