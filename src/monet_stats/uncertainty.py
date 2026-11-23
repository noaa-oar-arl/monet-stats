"""
Uncertainty quantification tools for model evaluation (Aero Protocol Compliant).
"""

from typing import Any, Callable, Dict, Union

import numpy as np
import xarray as xr

from .utils_stats import _update_history, ensure_single_chunk


def block_bootstrap(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    metric_func: Callable,
    n_boot: int = 1000,
    block_size: int = 1,
    dim: str = "time",
    axis: int = -1,
    confidence_level: float = 0.95,
    **metric_kwargs: Any,
) -> Union[Dict[str, float], xr.Dataset]:
    """
    Perform block-bootstrapping to calculate confidence intervals for a metric (Aero Protocol).

    Typical Use Cases
    -----------------
    - Calculating 95% confidence intervals for any performance metric (RMSE, Bias, etc.).
    - Accounting for temporal or spatial autocorrelation in the data.
    - Providing context for whether model improvements are statistically significant.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values.
    metric_func : Callable
        The metric function to evaluate (e.g., MAE, RMSE).
    n_boot : int, optional
        Number of bootstrap iterations, by default 1000.
    block_size : int, optional
        Size of the blocks for bootstrapping, by default 1 (standard bootstrap).
    dim : str, optional
        Dimension along which to bootstrap (xarray only).
    axis : int, optional
        Axis along which to bootstrap (numpy only).
    confidence_level : float, optional
        Confidence level for the intervals, by default 0.95.
    **metric_kwargs : Any
        Additional keyword arguments passed to the metric function.

    Returns
    -------
    dict or xarray.Dataset
        Mean, lower bound, and upper bound of the bootstrapped metric.
    """

    def _bootstrap_numpy(o: np.ndarray, m: np.ndarray) -> np.ndarray:
        n = o.shape[axis]
        n_blocks = n // block_size
        if n_blocks == 0:
            return np.array([np.nan] * n_boot)

        # Pre-calculate block indices
        block_indices = [np.arange(i, i + block_size) for i in range(0, n - block_size + 1)]
        n_available_blocks = len(block_indices)

        boot_metrics = []
        for _ in range(n_boot):
            # Sample blocks with replacement
            idx = np.random.choice(n_available_blocks, n_blocks)
            sample_idx = np.concatenate([block_indices[i] for i in idx])

            # Slice along the bootstrap axis
            o_boot = np.take(o, sample_idx, axis=axis)
            m_boot = np.take(m, sample_idx, axis=axis)

            boot_metrics.append(metric_func(o_boot, m_boot, **metric_kwargs))

        return np.array(boot_metrics)

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        obs = ensure_single_chunk(obs, dim)
        mod = ensure_single_chunk(mod, dim)

        res = xr.apply_ufunc(
            _bootstrap_numpy,
            obs,
            mod,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[["bootstrap"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {"bootstrap": n_boot}},
        )

        # Calculate percentiles for confidence intervals; skipna=True (default)
        # ensures any NaN bootstrap samples (e.g., from degenerate blocks) are ignored
        alpha = (1 - confidence_level) / 2
        res_mean = res.mean(dim="bootstrap", skipna=True)
        res_low = res.quantile(alpha, dim="bootstrap", skipna=True).drop_vars("quantile", errors="ignore")
        res_high = res.quantile(1 - alpha, dim="bootstrap", skipna=True).drop_vars("quantile", errors="ignore")

        ds = xr.Dataset({"mean": res_mean, "lower": res_low, "upper": res_high})
        return _update_history(ds, f"Block-Bootstrap ({metric_func.__name__}, n={n_boot})")

    # NumPy path
    boot_samples = _bootstrap_numpy(np.asarray(obs), np.asarray(mod))
    alpha = (1 - confidence_level) / 2
    return {
        "mean": float(np.nanmean(boot_samples)),
        "lower": float(np.nanquantile(boot_samples, alpha)),
        "upper": float(np.nanquantile(boot_samples, 1 - alpha)),
    }


def paired_block_bootstrap(
    obs: xr.DataArray,
    forecast: xr.DataArray,
    control: xr.DataArray,
    *,
    metric: Union[str, Callable] = "RMSE",
    n_boot: int = 1000,
    block_size: int = 1,
    dim: str = "time",
    confidence_level: float = 0.95,
    seed: Union[int, None] = None,
) -> xr.Dataset:
    """Estimate a percentile interval for a paired forecast-minus-control score.

    Use the same overlapping blocks of initialization dates for both models.
    Each resample contains floor(N / block_size) complete blocks. Non-finite
    triplets are omitted. A custom callable receives two finite NumPy vectors.
    """
    from .utils_stats import _reference_pairs

    if not 0 < confidence_level < 1 or n_boot < 1:
        raise ValueError("Require 0 < confidence_level < 1 and n_boot >= 1.")
    obs, forecast, control, _ = _reference_pairs(obs, forecast, control, None)
    n = obs.sizes[dim]
    if not 1 <= block_size <= n:
        raise ValueError("Require 1 <= block_size <= the number of samples.")
    if not callable(metric) and metric not in ("MB", "MAE", "MSE", "RMSE"):
        raise ValueError("metric must be MB, MAE, MSE, RMSE, or a callable.")
    if callable(metric) and n * n_boot > 10_000_000:
        raise ValueError("The requested custom-metric bootstrap exceeds 10 million sampled values per series.")

    rng = np.random.default_rng(seed)
    starts = rng.integers(n - block_size + 1, size=(n_boot, n // block_size))
    alpha = (1 - confidence_level) / 2
    count = obs.notnull().sum(dim)

    if callable(metric):

        def interval(o, f, c):
            def delta(indices):
                o_sample, f_sample, c_sample = o[indices], f[indices], c[indices]
                valid = np.isfinite(o_sample)
                if not valid.any():
                    return np.nan
                return metric(o_sample[valid], f_sample[valid]) - metric(o_sample[valid], c_sample[valid])

            original = delta(slice(None))
            differences = np.array([delta((start[:, None] + np.arange(block_size)).ravel()) for start in starts])
            low, high = np.nanquantile(differences, [alpha, 1 - alpha])
            return original, low, high

        original, lower, upper = xr.apply_ufunc(
            interval,
            *(ensure_single_chunk(field, dim) for field in (obs, forecast, control)),
            input_core_dims=[[dim]] * 3,
            output_core_dims=[[]] * 3,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float] * 3,
        )
    else:
        errors = xr.Dataset({"forecast": forecast - obs, "control": control - obs})
        if metric == "MAE":
            errors = abs(errors)
        elif metric in ("MSE", "RMSE"):
            errors = errors**2

        def finish(values):
            return np.sqrt(values) if metric == "RMSE" else values

        original = finish(errors.forecast.mean(dim)) - finish(errors.control.mean(dim))
        partials = errors.fillna(0).assign(count=obs.notnull().astype(float))
        totals = partials.rolling({dim: block_size}).sum().isel({dim: slice(block_size - 1, None)})

        def interval(f, c, n_valid):
            # Process resamples in bounded batches to avoid a bootstrap-by-time output array.
            differences = np.empty(n_boot)
            for begin in range(0, n_boot, 64):
                indices = starts[begin : begin + 64]
                denominator = n_valid[indices].sum(axis=-1)
                denominator = np.where(denominator > 0, denominator, np.nan)
                first = finish(f[indices].sum(axis=-1) / denominator)
                second = finish(c[indices].sum(axis=-1) / denominator)
                differences[begin : begin + len(indices)] = first - second
            return tuple(np.nanquantile(differences, [alpha, 1 - alpha]))

        lower, upper = xr.apply_ufunc(
            interval,
            *(ensure_single_chunk(totals[name], dim) for name in ("forecast", "control", "count")),
            input_core_dims=[[dim]] * 3,
            output_core_dims=[[]] * 2,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float] * 2,
        )
    result = xr.Dataset({"difference": original, "lower": lower, "upper": upper, "paired_count": count})
    return _update_history(result, "Paired block bootstrap")
