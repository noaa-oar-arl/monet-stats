"""Shared xarray scorecard construction."""

from functools import cache

import numpy as np

from . import contingency_metrics as categorical
from . import correlation_metrics as correlation
from . import error_metrics as error
from . import relative_metrics as relative
from .utils_stats import _prepare_pairs, _resolve_axis_to_dim, _update_history

_UNWEIGHTED = {
    "IOA": error.IOA,
    "MNB": error.MNB,
    "MNE": error.MNE,
    "NME": relative.NME,
    "NMSE": error.NMSE,
    "FAC2": error.FAC2,
}
_CATEGORICAL = {
    "POD": categorical._pod_from_counts,
    "FAR": categorical._far_from_counts,
    "HSS": categorical._hss_from_counts,
    "CSI": categorical._csi_from_counts,
    "TSS": categorical._tss_from_counts,
    "ETS": categorical._ets_from_counts,
    "FBI": categorical._fbi_from_counts,
    "BSS_binary": categorical._bss_binary_from_counts,
}
_COVERAGE = ("observation_count", "paired_count", "valid_weight")
_EVENT_SCORES = tuple(_CATEGORICAL)
VERIFY_METRICS = (
    *_COVERAGE,
    "MAE",
    "RMSE",
    "MSE",
    "MB",
    "R",
    "SpearmanR",
    "IOA",
    "NMB",
    "NME",
    "MNB",
    "MNE",
    "NSE",
    "R2",
    "FAC2",
)
SUMMARY_METRICS = (
    "N",
    *_COVERAGE,
    "Obs",
    "Mod",
    "MB",
    "MAE",
    "RMSE",
    "R",
    "IOA",
    "NMB",
    "MNB",
    "MNE",
    "NSE",
    "CRMSE",
    "MdnB",
    "KGE",
    "R2",
    "CCC",
    "NMSE",
    "FAC2",
    "MG",
    "VG",
)


def _build_scorecard(obs, mod, *, axis, weights, join, threshold, metrics, defaults, minval=None, maxval=None):
    dims = _resolve_axis_to_dim(obs, axis)
    obs, mod, weights, observed, paired = _prepare_pairs(
        obs, mod, join=join, weights=weights, minval=minval, maxval=maxval
    )
    chosen = (*defaults, *(_EVENT_SCORES if threshold is not None else ())) if metrics is None else metrics
    if isinstance(chosen, str):
        chosen = (chosen,)

    @cache
    def valid_weight():
        return paired.sum(dims) if weights is None else weights.where(paired, 0).sum(dims)

    def reduce_mean(field):
        # Centered moments below use two chunk-wise reductions and float64 arithmetic.
        if weights is None:
            return field.mean(dims, skipna=True)
        count = valid_weight()
        return (field * weights).sum(dims, skipna=True) / count.where(count > 0)

    @cache
    def obs_mean():
        return reduce_mean(obs.astype("float64"))

    @cache
    def mod_mean():
        return reduce_mean(mod.astype("float64"))

    @cache
    def difference():
        return mod.astype("float64") - obs.astype("float64")

    @cache
    def bias():
        return reduce_mean(difference())

    @cache
    def mse():
        return reduce_mean(difference() ** 2)

    @cache
    def centered_obs():
        return obs.astype("float64") - obs_mean()

    @cache
    def centered_mod():
        return mod.astype("float64") - mod_mean()

    @cache
    def obs_variance():
        return reduce_mean(centered_obs() ** 2)

    @cache
    def mod_variance():
        return reduce_mean(centered_mod() ** 2)

    @cache
    def covariance():
        return reduce_mean(centered_obs() * centered_mod())

    @cache
    def pearson():
        sd_product = np.sqrt(obs_variance() * mod_variance())
        return (covariance() / sd_product.where(sd_product > 0)).clip(-1, 1)

    def concordance():
        scale = obs_variance() + mod_variance() + (mod_mean() - obs_mean()) ** 2
        return 2 * covariance() / scale.where(scale > 0)

    def kge():
        alpha = np.sqrt(mod_variance() / obs_variance().where(obs_variance() > 0))
        beta = mod_mean() / obs_mean().where(obs_mean() != 0)
        return 1 - np.sqrt((pearson() - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    @cache
    def counts():
        if threshold is None:
            raise ValueError("Categorical scores require a threshold.")
        return categorical._contingency_table(obs, mod, threshold, axis=dims)

    def nmb():
        denominator = obs_mean()
        return 100 * bias() / denominator.where(denominator != 0)

    functions = {
        "N": lambda: observed.sum(dims),
        "observation_count": lambda: observed.sum(dims),
        "paired_count": lambda: paired.sum(dims),
        "valid_weight": valid_weight,
        "Obs": obs_mean,
        "Mod": mod_mean,
        "MB": bias,
        "MAE": lambda: reduce_mean(abs(difference())),
        "MSE": mse,
        "RMSE": lambda: np.sqrt(mse()),
        "CRMSE": lambda: np.sqrt(reduce_mean((difference() - bias()) ** 2)),
        "R": pearson,
        "R2": lambda: pearson() ** 2,
        "CCC": concordance,
        "KGE": kge,
        "NSE": lambda: 1 - mse() / obs_variance().where(obs_variance() > 0),
        "NMB": nmb,
        "MdnB": lambda: error.MdnB(obs, mod, axis=dims),
        "SpearmanR": lambda: correlation.spearmanr(obs, mod, axis=dims),
        "MG": lambda: relative.MG(obs, mod, axis=dims, weights=weights),
        "VG": lambda: relative.VG(obs, mod, axis=dims, weights=weights),
    }
    result = {}
    for name in chosen:
        if name in _CATEGORICAL:
            result[name] = _CATEGORICAL[name](obs, *counts())
        elif name in _UNWEIGHTED:
            result[name] = _UNWEIGHTED[name](obs, mod, axis=dims)
        elif name in functions:
            value = functions[name]()
            if name not in {"MdnB", "SpearmanR", "MG", "VG"}:
                value = _update_history(value.assign_attrs(obs.attrs), name)
            result[name] = value
        else:
            raise ValueError(f"Unknown verification metric {name!r}.")
    return result, obs, mod, dims
