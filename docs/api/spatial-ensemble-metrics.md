---
type: reference
category: reference
tags: [spatial, ensemble, crps]
---

# Spatial & Ensemble Metrics

Spatial verification and ensemble analysis metrics.

::: monet_stats.spatial_ensemble_metrics

## Reference skill and spread

Use `forecast.monet_stats.crps(obs, dim="member", fair=True)` to apply the
fair estimator. Cases with missing members are excluded. Fair CRPS
requires at least two members. Use `forecast.monet_stats.crpss(obs,
reference, dim=["time", "cell"], weights=cell_area)` to score aggregate
CRPS skill against a reference ensemble or deterministic forecast on
the same valid samples. A zero reference score yields NaN.

`forecast.monet_stats.spread_rmse()` returns RMS ensemble spread,
ensemble-mean RMSE, and paired count on common complete cases.
`spread_error()` still returns mean spread and mean absolute error.
`rank_histogram()` splits ties evenly across admissible ranks and
excludes incomplete cases. BSS uses raw probabilities and supports a
reference forecast. Use `BSS_binary()` for a thresholded deterministic
forecast.
