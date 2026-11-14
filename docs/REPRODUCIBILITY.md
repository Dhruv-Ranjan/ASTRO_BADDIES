# Reproducibility

## Inference

Install `FINAL_PROJECT_NASA/requirements.txt` and
`backend/requirements-research.txt`, then run:

```powershell
.\.venv\Scripts\python.exe .\backend\mission_api.py
```

The release includes the selected production checkpoints below 100 MB per
file. The large sequence tensor dataset is not included.

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

