"""
Spatial Skill Metrics for Model Evaluation (Aero Protocol Compliant)
"""

from typing import Iterable, Optional, Union

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from .utils_stats import _resolve_axis_to_dim, _update_history


def _uniform_filter(data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a uniform filter to the data.

    Parameters
    ----------
    data : np.ndarray
        The input data array.
    window_size : int
        The size of the filter window.

    Returns
    -------
    np.ndarray
        The filtered data.
    """
    return uniform_filter(data.astype(float), size=window_size)


def _window_fractions(obs, mod, dims, size, threshold, boundary, min_coverage):
    if boundary not in ("valid", "periodic"):
        raise ValueError("boundary must be 'valid' or 'periodic'.")
    if size < 1 or not 0 < min_coverage <= 1:
        raise ValueError("Require a positive window size and 0 < min_coverage <= 1.")
    if not dims or len(set(dims)) != len(dims):
        raise ValueError("Supply distinct spatial dimensions.")
    obs, mod = xr.align(obs, mod, join="inner", copy=False)
    valid = np.isfinite(obs) & np.isfinite(mod)
    if threshold is None:
        threshold = obs.where(valid).mean(dims)
    event_obs = (obs >= threshold) & valid
    event_mod = (mod >= threshold) & valid
    fields = xr.Dataset({"obs": event_obs, "mod": event_mod, "valid": valid})
    left = size // 2
    right = size - 1 - left
    if boundary == "periodic":
        longitude = dims[-1]
        fields = fields.pad({longitude: (left, right)}, mode="wrap")
    windows = dict.fromkeys(dims, size)
    totals = fields.rolling(windows, center=True).sum()
    if boundary == "periodic":
        totals = totals.isel({longitude: slice(left, left + obs.sizes[longitude])})
    denominator = totals.valid.where(totals.valid >= min_coverage * size ** len(dims))
    return totals.obs / denominator, totals.mod / denominator


def FSS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    window_size: int = 3,
    threshold: Optional[float] = None,
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    *,
    boundary: Optional[str] = None,
    min_coverage: float = 1.0,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Fractions Skill Score (FSS) (Aero Protocol).

    Typical Use Cases
    -----------------
    - Evaluating the spatial skill of a forecast, particularly for high-resolution models.
    - Assessing forecast performance at different spatial scales by varying the window size.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values.
    window_size : int, optional
        The size of the square window for calculating fractions, by default 3.
    threshold : float, optional
        The threshold to define an event. If None, uses mean of observations (Lazy-friendly).
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the fractions (xarray only).
        If None, uses all dimensions.
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the fractions.
    boundary : {"valid", "periodic"}, optional
        Use common paired-window semantics. Periodic wraps the last
        spatial dimension. None preserves each backend's historical edge rules.
    min_coverage : float, optional
        Minimum fraction of finite pairs in a complete window when boundary is set.

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The Fractions Skill Score, ranging from 0 (no skill) to 1 (perfect skill).

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> obs = xr.DataArray(np.random.rand(10, 10), dims=['lat', 'lon'])
    >>> mod = xr.DataArray(np.random.rand(10, 10), dims=['lat', 'lon'])
    >>> score = FSS(obs, mod, window_size=3, threshold=0.5)
    """
    if boundary is not None:
        labeled = isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray)
        if not labeled:
            obs, mod = xr.DataArray(obs), xr.DataArray(mod)
        dims = _resolve_axis_to_dim(obs, dim if dim is not None else axis)
        dims = [dims] if isinstance(dims, str) else list(dims)
        obs_frac, mod_frac = _window_fractions(obs, mod, dims, window_size, threshold, boundary, min_coverage)
        numerator = ((obs_frac - mod_frac) ** 2).sum(dims, min_count=1)
        denominator = (obs_frac**2 + mod_frac**2).sum(dims, min_count=1)
        result = 1.0 - numerator / denominator.where(denominator > 0)
        if labeled:
            return _update_history(result, "Fractions Skill Score (FSS)")
        return result.item() if result.ndim == 0 else result.to_numpy()

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")

        valid = np.isfinite(obs) & np.isfinite(mod)
        if threshold is None:
            # Use xarray's native mean to maintain laziness
            threshold = obs.where(valid).mean()

        obs_binary = (obs >= threshold).where(valid)
        mod_binary = (mod >= threshold).where(valid)

        # Rolling mean for fractions
        # Resolve dimensions using Aero Protocol utility
        roll_dims = _resolve_axis_to_dim(obs, dim if dim is not None else axis)

        if isinstance(roll_dims, str):
            roll_dims = [roll_dims]

        roll_kwargs = {d: window_size for d in roll_dims}
        obs_frac = obs_binary.rolling(dim=roll_kwargs, center=True).mean()
        mod_frac = mod_binary.rolling(dim=roll_kwargs, center=True).mean()

        mse = ((obs_frac - mod_frac) ** 2).mean(dim=roll_dims)
        mse_ref = (obs_frac**2).mean(dim=roll_dims) + (mod_frac**2).mean(dim=roll_dims)

        res = xr.where(mse_ref == 0, 1.0, 1 - (mse / mse_ref))
        return _update_history(res, "Fractions Skill Score (FSS)")

    # NumPy path
    obs_arr = np.asarray(obs)
    mod_arr = np.asarray(mod)

    valid = np.isfinite(obs_arr) & np.isfinite(mod_arr)
    if threshold is None:
        threshold = np.nanmean(np.where(valid, obs_arr, np.nan))

    obs_binary = ((obs_arr >= threshold) & valid).astype(float)
    mod_binary = ((mod_arr >= threshold) & valid).astype(float)

    available = _uniform_filter(valid, window_size)
    denominator = np.where(available > 0, available, np.nan)
    obs_frac = _uniform_filter(obs_binary, window_size) / denominator
    mod_frac = _uniform_filter(mod_binary, window_size) / denominator

    # Use specified axis for mean calculation if provided
    mse = np.nanmean((obs_frac - mod_frac) ** 2, axis=axis)
    mse_ref = np.nanmean(obs_frac**2, axis=axis) + np.nanmean(mod_frac**2, axis=axis)

    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(mse_ref == 0, 1.0, 1 - (mse / mse_ref))
        return res.item() if np.ndim(res) == 0 else res


def VETS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Volumetric Equitable Threat Score (VETS) (Aero Protocol).

    Typical Use Cases
    -----------------
    - Evaluating the skill of volumetric forecasts, such as precipitation accumulation.
    - Provides an equitable score that accounts for random chance.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the score (xarray only).
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the score.

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The Volumetric Equitable Threat Score.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")

        # Resolve reduction dimensions
        reduction_dim = _resolve_axis_to_dim(obs, dim if dim is not None else axis)

        # Use xr.where for minimum/maximum to be safer with dask
        hits = xr.where(obs < mod, obs, mod).sum(dim=reduction_dim)
        sum_obs = obs.sum(dim=reduction_dim)
        sum_mod = mod.sum(dim=reduction_dim)
        misses = sum_obs - hits
        false_alarms = sum_mod - hits
        total_union = xr.where(obs > mod, obs, mod).sum(dim=reduction_dim)

        hits_random = (sum_obs * sum_mod) / total_union
        denominator = hits + misses + false_alarms - hits_random

        res = xr.where(denominator == 0, 1.0, (hits - hits_random) / denominator)
        return _update_history(res, "Volumetric Equitable Threat Score (VETS)")

    # NumPy path
    from .utils_stats import _nanmask_inputs

    o_, m_ = _nanmask_inputs(obs, mod)
    obs_arr = o_.filled(np.nan)
    mod_arr = m_.filled(np.nan)

    hits = np.nansum(np.minimum(obs_arr, mod_arr), axis=axis)
    sum_obs = np.nansum(obs_arr, axis=axis)
    sum_mod = np.nansum(mod_arr, axis=axis)
    misses = sum_obs - hits
    false_alarms = sum_mod - hits
    total_union = sum_obs + sum_mod - hits

    hits_random = (sum_obs * sum_mod) / total_union
    denominator = hits + misses + false_alarms - hits_random

    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(denominator == 0, 1.0, (hits - hits_random) / denominator)
        return res.item() if np.ndim(res) == 0 else res


def multiscale_fss(
    obs: xr.DataArray,
    mod: xr.DataArray,
    *,
    windows: Iterable[int],
    thresholds: Iterable[float],
    dim: Union[str, Iterable[str]],
    boundary: str = "valid",
    min_coverage: float = 1.0,
) -> xr.DataArray:
    """Return FSS by window size and threshold for structured-grid neighborhoods.

    Periodic boundaries wrap the final spatial dimension only.
    """
    windows, thresholds = list(windows), list(thresholds)
    if not windows or not thresholds or len(set(windows)) != len(windows) or len(set(thresholds)) != len(thresholds):
        raise ValueError("Supply nonempty, unique windows and thresholds.")
    if not np.isfinite(thresholds).all() or any(
        not isinstance(size, (int, np.integer)) or size < 1 for size in windows
    ):
        raise ValueError("Require finite thresholds and positive integer windows.")
    if {"window", "threshold"} & (set(obs.dims) | set(mod.dims)):
        raise ValueError("The 'window' and 'threshold' dimensions are reserved for the result.")
    spatial_dims = [dim] if isinstance(dim, str) else list(dim)
    threshold = xr.DataArray(thresholds, coords={"threshold": thresholds}, dims="threshold")
    scores = [
        FSS(
            obs,
            mod,
            window_size=size,
            threshold=threshold,
            dim=spatial_dims,
            boundary=boundary,
            min_coverage=min_coverage,
        )
        for size in windows
    ]
    return _update_history(xr.concat(scores, dim=xr.IndexVariable("window", windows)), "Multiscale FSS")
