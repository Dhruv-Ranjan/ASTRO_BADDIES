# Testing

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests
```

The focused tests cover the database helper, mission planner, and integrated
pipeline. The legacy inference-validation and end-to-end scripts are excluded
from automatic collection because they execute immediately and require the
excluded tensor dataset; run them directly after restoring inputs. Such
requirements are not hidden successes.

```powershell
.\.venv\Scripts\python.exe .\FINAL_PROJECT_NASA\src\test_end_to_end.py
.\.venv\Scripts\python.exe .\FINAL_PROJECT_NASA\src\test_inference_validation.py
```

Before a submission, also verify:

- the frontend opens from `frontend/`;
- `/health` responds when the local API is running;
- the repository sample reconstructs six bands;
- no `.env`, cache, database, tensor dataset, or bytecode is staged;
- reported JSON metrics match the accompanying run artifacts.

