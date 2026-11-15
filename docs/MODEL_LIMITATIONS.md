# Model limitations

- Reported validation data are not uniformly geographically disjoint.
- Drought supervision is largely proxy-derived, not field-verified ground
  truth.
- The separate NASA POWER forecasting workflow uses gridded/reanalysis-informed
  climate context. Its future 30-day rainfall-deficit target is a
  POWER-derived meteorological proxy; it is not independent soil-moisture,
  crop-impact, or field-station truth.
- POWER rolling features include the matched calendar-day row, so lag daily
  inputs before claiming that forecasts made at an intraday Sentinel-2
  timestamp use only information already available at that time.
- The six-band reconstruction is sensitive to normalization and domain shift.
- Confidence values are not calibrated probabilities.
- The mission planner uses estimated access windows, cloud heuristics, and
  resource costs; it is not an orbital dynamics or flight-software product.
- Exact retraining requires excluded tensor data, preprocessing, seeds, and
  compatible dependencies.
- Outputs must not be used as the sole basis for agricultural, emergency, or
  policy decisions.

