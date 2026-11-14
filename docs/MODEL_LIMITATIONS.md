# Model limitations

- Reported validation data are not uniformly geographically disjoint.
- Drought supervision is largely proxy-derived, not field-verified ground
  truth.
- The six-band reconstruction is sensitive to normalization and domain shift.
- Confidence values are not calibrated probabilities.
- The mission planner uses estimated access windows, cloud heuristics, and
  resource costs; it is not an orbital dynamics or flight-software product.
- Exact retraining requires excluded tensor data, preprocessing, seeds, and
  compatible dependencies.
- Outputs must not be used as the sole basis for agricultural, emergency, or
  policy decisions.

