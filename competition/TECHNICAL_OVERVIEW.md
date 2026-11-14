# Technical overview

The system combines a `SpectralNetV10` RGB-to-six-band model, temporal drought
forecasting code, and the AURORA resource-aware planner. The planner consumes
forecast rows, ranks candidate regions, selects a feasible target, and records
a simulated belief update. The frontend is static and the local Python API
provides optional checkpoint-backed inference.

