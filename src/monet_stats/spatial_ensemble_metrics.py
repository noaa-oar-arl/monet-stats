"""
Spatial and Ensemble Metrics for Atmospheric Sciences (Aero Protocol Compliant)
"""

from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import xarray as xr

from .utils_stats import _resolve_axis_to_dim, _skill_score, _update_history


def EDS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    threshold: float,
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, Iterable[int]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Extreme Dependency Score (EDS) for rare event detection (Aero Protocol).

    Typical Use Cases
    -----------------
    - Assessing model performance for rare extreme events (e.g., heavy precipitation).
    - Used when traditional scores like CSI or ETS go to zero as the event becomes rarer.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed field.
    mod : xarray.DataArray or numpy.ndarray
        Model field.
    threshold : float
        Event threshold to define the extreme event.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the score (xarray only).
        If None, reduces over all dimensions.
    axis : int or iterable of int, optional
        Axis or axes along which to compute the score (numpy only).

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        Extreme Dependency Score.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.zeros((10, 10)); obs[5, 5] = 1
    >>> mod = np.zeros((10, 10)); mod[5, 5] = 1
    >>> EDS(obs, mod, threshold=0.5)
    1.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        obs_bin = obs >= threshold
        mod_bin = mod >= threshold

        mask = obs.notnull() & mod.notnull()
        obs_bin = obs_bin & mask
        mod_bin = mod_bin & mask

        hits = (obs_bin & mod_bin).sum(dim=dim)
        n_obs = obs_bin.sum(dim=dim)
        n_mod = mod_bin.sum(dim=dim)
        n = mask.sum(dim=dim)

        # EDS = log(hits/n) / log(n_obs*n_mod / n^2)
        # Avoid log(0) and division by zero
        # p = n_obs / n, q = n_mod / n
        with np.errstate(divide="ignore", invalid="ignore"):
            eds = xr.where(
                (hits > 0) & (n_obs > 0) & (n_mod > 0), np.log(hits / n) / np.log((n_obs / n) * (n_mod / n)), np.nan
            )

        return _update_history(eds, "Extreme Dependency Score (EDS)")

    # NumPy path
    obs_arr = np.asarray(obs)
    mod_arr = np.asarray(mod)
    mask = ~np.isnan(obs_arr) & ~np.isnan(mod_arr)

    obs_bin = (obs_arr >= threshold) & mask
    mod_bin = (mod_arr >= threshold) & mask

    hits = np.sum(obs_bin & mod_bin, axis=axis)
    n_obs = np.sum(obs_bin, axis=axis)
    n_mod = np.sum(mod_bin, axis=axis)
    n = np.sum(mask, axis=axis)

    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(
            (hits > 0) & (n_obs > 0) & (n_mod > 0), np.log(hits / n) / np.log((n_obs / n) * (n_mod / n)), np.nan
        )
        return res.item() if np.ndim(res) == 0 else res


def CRPS(
    ensemble: Union[xr.DataArray, np.ndarray],
    obs: Union[xr.DataArray, np.ndarray],
    axis: Union[int, str] = 0,
    *,
    fair: bool = False,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Continuous Ranked Probability Score (CRPS) for ensemble forecasts.

    Supports lazy evaluation via Xarray/Dask.

    Parameters
    ----------
    ensemble : xarray.DataArray or numpy.ndarray
        Ensemble forecasts. If DataArray, should have an ensemble dimension.
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    fair : bool, optional
        Use the fair estimator. Cases with fewer than two members return NaN.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        CRPS values.

    Examples
    --------
    >>> import numpy as np
    >>> ens = np.array([[1, 2], [2, 3], [3, 4]])
    >>> obs = np.array([2, 3])
    >>> CRPS(ens, obs, axis=0)
    array([0.22222222, 0.22222222])
    """

    def _crps_numpy(ens: np.ndarray, observation: np.ndarray, ens_axis: int = 0) -> np.ndarray:
        """
        Core NumPy implementation of CRPS using the energy form.
        CRPS = E|X-y| - 0.5 * E|X-X'|
        """
        ens_axis %= ens.ndim
        n = ens.shape[ens_axis]
        if n == 0 or (fair and n < 2):
            return np.full(
                np.broadcast_shapes(ens.shape[:ens_axis] + ens.shape[ens_axis + 1 :], observation.shape), np.nan
            )

        # 1. Mean absolute error part: E|X-y|
        obs_broadcast = np.expand_dims(observation, axis=ens_axis)
        mae = np.mean(np.abs(ens - obs_broadcast), axis=ens_axis)

        # 2. Ensemble spread part: 0.5 * E|X-X'|
        # Optimized O(N log N) spread calculation using sorted weights
        ens_sorted = np.sort(ens, axis=ens_axis)
        i = np.arange(1, n + 1)
        weights = (2 * i - n - 1) / (n * (n - 1 if fair else n))

        # Reshape weights for broadcasting
        w_shape = [1] * ens.ndim
        w_shape[ens_axis] = n
        weights = weights.reshape(w_shape)

        spread = np.sum(weights * ens_sorted, axis=ens_axis)

        valid = np.isfinite(ens).all(axis=ens_axis) & np.isfinite(observation)
        return np.where(valid, np.maximum(mae - spread, 0), np.nan)

    if isinstance(ensemble, xr.DataArray) and isinstance(obs, xr.DataArray):
        # Determine core dimension
        if isinstance(axis, int):
            ens_dim = ensemble.dims[axis]
        else:
            ens_dim = axis

        if ens_dim in obs.dims:
            raise ValueError("Observations must not include the member dimension.")
        ensemble, obs = xr.align(ensemble, obs, join="inner", copy=False)
        if ensemble.chunks is not None or obs.chunks is not None:
            ensemble, obs = xr.broadcast(ensemble, obs, exclude=[ens_dim])
            existing = ensemble.chunksizes if ensemble.chunks is not None else obs.chunksizes
            outer = {name: max(chunks) for name, chunks in existing.items() if name != ens_dim}
            outer.update({name: obs.sizes[name] for name in obs.dims if name not in outer})
            budget = max(1, 32 * 1024**2 // (8 * max(1, ensemble.sizes[ens_dim])))
            for name in sorted(outer, key=outer.get, reverse=True):
                other = max(1, int(np.prod([size for key, size in outer.items() if key != name])))
                outer[name] = max(1, min(outer[name], budget // other))
            ensemble = ensemble.chunk({**outer, ens_dim: -1})
            obs = obs.chunk(outer)

        res = xr.apply_ufunc(
            _crps_numpy,
            ensemble,
            obs,
            input_core_dims=[[ens_dim], []],
            output_core_dims=[[]],
            kwargs={"ens_axis": -1},
            dask="parallelized",
            output_dtypes=[float],
        )
        return _update_history(res, "Continuous Ranked Probability Score (CRPS)")

    return _crps_numpy(np.asarray(ensemble), np.asarray(obs), ens_axis=axis)


def CRPSS(
    ensemble: xr.DataArray,
    obs: xr.DataArray,
    reference: xr.DataArray,
    *,
    member_dim: str = "member",
    reference_member_dim: Optional[str] = None,
    dim=None,
    weights=None,
    fair: bool = False,
) -> xr.DataArray:
    """Compare ensemble CRPS to a reference ensemble or deterministic forecast."""
    from .utils_stats import _prepare_pairs

    forecast_score = CRPS(ensemble, obs, axis=member_dim, fair=fair)
    ref_dim = reference_member_dim or member_dim
    if ref_dim in reference.dims:
        reference_score = CRPS(reference, obs, axis=ref_dim, fair=fair)
    else:
        reference_score = abs(reference - obs)
    forecast_score, reference_score, weights, _, _ = _prepare_pairs(forecast_score, reference_score, weights=weights)
    if weights is not None:
        forecast_score, reference_score = forecast_score * weights, reference_score * weights
    skill = _skill_score(forecast_score.sum(dim, min_count=1), reference_score.sum(dim, min_count=1))
    return _update_history(skill, "Continuous Ranked Probability Skill Score")


def spread_rmse(
    ensemble: xr.DataArray, obs: xr.DataArray, *, member_dim: str = "member", dim=None, weights=None
) -> xr.Dataset:
    """Return RMS ensemble spread and ensemble-mean RMSE on common complete cases."""
    from .utils_stats import _prepare_pairs

    if member_dim in obs.dims:
        raise ValueError("Observations must not include the member dimension.")
    ensemble, obs = xr.align(ensemble.astype("float64"), obs, join="inner", copy=False)
    ensemble = ensemble.where(np.isfinite(ensemble))
    mean = ensemble.mean(member_dim, skipna=False)
    obs, mean, weights, _, paired = _prepare_pairs(obs, mean, weights=weights)
    variance = ensemble.var(member_dim, skipna=False).where(mean.notnull())
    squared_error = (mean - obs) ** 2
    if weights is not None:
        variance = variance.weighted(weights).mean(dim)
        squared_error = squared_error.weighted(weights).mean(dim)
    else:
        variance = variance.mean(dim)
        squared_error = squared_error.mean(dim)
    result = xr.Dataset({"spread": np.sqrt(variance), "rmse": np.sqrt(squared_error), "paired_count": paired.sum(dim)})
    return _update_history(result, "Ensemble spread and RMSE")


def spread_error(
    ensemble: Union[xr.DataArray, np.ndarray],
    obs: Union[xr.DataArray, np.ndarray],
    axis: Union[int, str] = 0,
    dim: Optional[Union[str, Iterable[str]]] = None,
    reduce_axis: Optional[Union[int, Iterable[int]]] = None,
) -> Tuple[Any, Any]:
    """
    Spread-Error Relationship for ensemble forecasts (Aero Protocol).

    Typical Use Cases
    -----------------
    - Assessing if the ensemble spread is a good proxy for the forecast error.
    - Ideally, mean spread should equal RMSE of the ensemble mean.

    Parameters
    ----------
    ensemble : xarray.DataArray or numpy.ndarray
        Ensemble forecasts.
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the mean spread and error (xarray only).
        If None, reduces over all dimensions.
    reduce_axis : int or iterable of int, optional
        Axis or axes along which to compute the mean spread and error (numpy only).

    Returns
    -------
    mean_spread : float, numpy.ndarray, or xarray.DataArray
        Mean ensemble spread.
    mean_error : float, numpy.ndarray, or xarray.DataArray
        Mean absolute error of ensemble mean vs. obs.
    """
    if isinstance(ensemble, xr.DataArray) and isinstance(obs, xr.DataArray):
        # Resolve ensemble dimension
        e_dim = axis
        if isinstance(axis, int):
            e_dim = ensemble.dims[axis]

        spread_field = ensemble.std(dim=e_dim)
        ens_mean_field = ensemble.mean(dim=e_dim)
        error_field = abs(ens_mean_field - obs)

        # Average over specified dimensions (or all remaining)
        m_spread = spread_field.mean(dim=dim)
        m_error = error_field.mean(dim=dim)

        return _update_history(m_spread, "Mean Ensemble Spread"), _update_history(m_error, "Mean Ensemble Error")

    # NumPy path
    ens = np.asarray(ensemble, dtype=np.result_type(ensemble, "f4"))
    observation = np.asarray(obs, dtype=np.result_type(obs, "f4"))

    # Mask NaNs in observation across ensemble axis (broadcast mask)
    obs_nan_mask = np.isnan(observation)
    ens_nan_mask = np.any(np.isnan(ens), axis=axis)
    combined_mask = obs_nan_mask | ens_nan_mask

    # Apply mask before std/mean to avoid NaN contamination
    ens_m = np.where(np.expand_dims(combined_mask, axis=axis), np.nan, ens)
    obs_m = np.where(combined_mask, np.nan, observation)

    # Calculate spread and ensemble mean
    spread_f = np.nanstd(ens_m, axis=axis)
    ens_m_f = np.nanmean(ens_m, axis=axis)
    error_f = np.abs(ens_m_f - obs_m)

    # Average over specified axes
    m_spread = np.nanmean(spread_f, axis=reduce_axis)
    m_error = np.nanmean(error_f, axis=reduce_axis)

    if np.ndim(m_spread) == 0:
        return float(m_spread), float(m_error)
    return m_spread, m_error


def _valid_probability(probability):
    return np.isfinite(probability) & (probability >= 0) & (probability <= 1)


def BSS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    threshold: float = 0.5,
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, Iterable[int]]] = None,
    *,
    reference: Optional[Union[xr.DataArray, np.ndarray, float]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """Score raw forecast probabilities against a reference or sample climatology.

    The threshold applies to observations. Exclude invalid probabilities and
    missing observations from both scores. Zero reference error yields NaN.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        if reference is None or np.isscalar(reference):
            obs, mod = xr.align(obs, mod, join="inner", copy=False)
        else:
            reference = reference if isinstance(reference, xr.DataArray) else xr.DataArray(reference, dims=obs.dims)
            obs, mod, reference = xr.align(obs, mod, reference, join="inner", copy=False)
        reduction_dim = _resolve_axis_to_dim(obs, dim if dim is not None else axis)
        valid = np.isfinite(obs) & _valid_probability(mod)
        if reference is not None:
            valid = valid & _valid_probability(reference)
        event = (obs >= threshold).where(valid)
        prob = mod.where(valid)
        ref = event.mean(reduction_dim) if reference is None else reference
        skill = _skill_score(((prob - event) ** 2).mean(reduction_dim), ((ref - event) ** 2).mean(reduction_dim))
        return _update_history(skill, "Brier Skill Score (BSS)")

    obs, mod = np.asarray(obs), np.asarray(mod)
    valid = np.isfinite(obs) & _valid_probability(mod)
    if reference is not None:
        reference = np.asarray(reference)
        valid &= _valid_probability(reference)
    event = np.where(valid, obs >= threshold, np.nan)
    mod = np.where(valid, mod, np.nan)
    ref = np.nanmean(event, axis=axis, keepdims=True) if reference is None else reference
    bs = np.nanmean((mod - event) ** 2, axis=axis)
    bs_ref = np.nanmean((ref - event) ** 2, axis=axis)
    result = 1.0 - bs / np.where(bs_ref > 0, bs_ref, np.nan)
    return result.item() if np.ndim(result) == 0 else result


def _sal_numpy(
    obs_2d: np.ndarray,
    mod_2d: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Core NumPy implementation of SAL for a 2D field.

    Parameters
    ----------
    obs_2d : numpy.ndarray
        Observed 2D field.
    mod_2d : numpy.ndarray
        Model 2D field.
    threshold : float, optional
        Threshold for object identification.

    Returns
    -------
    S, A, L : float
        Structure, Amplitude, and Location components.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.random.rand(10, 10)
    >>> mod = np.random.rand(10, 10)
    >>> _sal_numpy(obs, mod)
    """
    import scipy.ndimage as ndi

    if threshold is None:
        threshold = np.nanmean(obs_2d)

    # Amplitude
    m_mod = np.nanmean(mod_2d)
    m_obs = np.nanmean(obs_2d)
    denom_a = m_mod + m_obs
    A = 2 * (m_mod - m_obs) / denom_a if denom_a != 0 else 0.0

    # Structure
    def structure(X):
        mask = threshold <= X
        if not np.any(mask):
            return 0.0, 0.0
        labeled, n = ndi.label(mask)
        masses = ndi.sum(X, labeled, index=np.arange(1, n + 1))
        max_mass = np.max(masses)
        total_mass = np.sum(masses)
        return max_mass, total_mass

    max_mod, sum_mod = structure(mod_2d)
    max_obs, sum_obs = structure(obs_2d)
    denom_s = (max_mod / sum_mod + max_obs / sum_obs) if sum_mod > 0 and sum_obs > 0 else 0
    S = 2 * (max_mod / sum_mod - max_obs / sum_obs) / denom_s if denom_s != 0 else 0.0

    # Location
    def centroid_and_spread(X):
        mask = threshold <= X
        if not np.any(mask):
            return np.array([0.0, 0.0]), 0.0, 0.0
        labeled, n = ndi.label(mask)
        centers = np.array(ndi.center_of_mass(X, labeled, index=np.arange(1, n + 1)))
        masses = ndi.sum(X, labeled, index=np.arange(1, n + 1))
        # Total mass-weighted center of the field's objects
        c = np.average(centers, axis=0, weights=masses)
        # Spread
        r = np.average(np.linalg.norm(centers - c, axis=1), weights=masses)
        return c, masses, r

    c_mod, _, r_mod = centroid_and_spread(mod_2d)
    c_obs, _, r_obs = centroid_and_spread(obs_2d)

    dist = np.linalg.norm(c_mod - c_obs)
    max_dist = np.sqrt(obs_2d.shape[0] ** 2 + obs_2d.shape[1] ** 2)
    L1 = dist / max_dist if max_dist != 0 else 0.0
    L2 = abs(r_mod - r_obs) / max_dist if max_dist != 0 else 0.0
    L = L1 + L2
    return float(S), float(A), float(L)


def SAL(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    threshold: Optional[float] = None,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> Union[
    Tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[float, float, float],
]:
    """
    Structure-Amplitude-Location (SAL) score for spatial verification (Aero Protocol).

    This implementation is vectorized over non-spatial dimensions using `xarray.apply_ufunc`
    and supports lazy evaluation for dimensions other than the spatial ones.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed field. Should be 2D (lat, lon) or multi-dimensional.
    mod : xarray.DataArray or numpy.ndarray
        Model field.
    threshold : float, optional
        Threshold for object identification. If None, uses mean of observations
        per slice (Lazy-friendly).
    lat_dim : str, optional
        Name of the latitude dimension. Default is 'lat'.
    lon_dim : str, optional
        Name of the longitude dimension. Default is 'lon'.

    Returns
    -------
    S, A, L : xarray.DataArray, numpy.ndarray, or float
        Structure, Amplitude, and Location components.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> obs = xr.DataArray(np.random.rand(10, 10, 10), dims=['time', 'lat', 'lon'])
    >>> mod = xr.DataArray(np.random.rand(10, 10, 10), dims=['time', 'lat', 'lon'])
    >>> S, A, L = SAL(obs, mod)
    """
    is_xr = isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray)

    if is_xr:
        obs, mod = xr.align(obs, mod, join="inner")
        # Determine spatial dimensions
        if lat_dim not in obs.dims or lon_dim not in obs.dims:
            # Fallback to last two dimensions if lat/lon not found
            spatial_dims = [obs.dims[-2], obs.dims[-1]]
        else:
            spatial_dims = [lat_dim, lon_dim]
    else:
        # For numpy, assume last two dimensions are spatial
        spatial_dims = [-2, -1]

    def _sal_wrapper(o: np.ndarray, m: np.ndarray, t: Optional[float]) -> Tuple[float, float, float]:
        """
        Wrapper for _sal_numpy to be used with xarray.apply_ufunc.

        Parameters
        ----------
        o : numpy.ndarray
            Observed field slice.
        m : numpy.ndarray
            Model field slice.
        t : float, optional
            Threshold.

        Returns
        -------
        Tuple[float, float, float]
            S, A, L components.

        Examples
        --------
        >>> import numpy as np
        >>> o = np.random.rand(10, 10)
        >>> m = np.random.rand(10, 10)
        >>> _sal_wrapper(o, m, 0.5)
        """
        return _sal_numpy(o, m, t)

    res_s, res_a, res_l = xr.apply_ufunc(
        _sal_wrapper,
        obs,
        mod,
        threshold,
        input_core_dims=[spatial_dims, spatial_dims, []],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )

    if is_xr:
        res_s = _update_history(res_s, "SAL: Structure component")
        res_a = _update_history(res_a, "SAL: Amplitude component")
        res_l = _update_history(res_l, "SAL: Location component")

    return res_s, res_a, res_l


def ensemble_mean(
    ensemble: Union[xr.DataArray, np.ndarray],
    axis: Union[int, str] = 0,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the ensemble mean.

    Parameters
    ----------
    ensemble : xarray.DataArray or numpy.ndarray
        Ensemble forecasts.
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Ensemble mean.
    """
    if isinstance(ensemble, xr.DataArray):
        dim = axis
        if isinstance(axis, int):
            dim = ensemble.dims[axis]
        res = ensemble.mean(dim=dim)
        return _update_history(res, "Ensemble Mean")
    return np.mean(ensemble, axis=axis)


def ensemble_std(
    ensemble: Union[xr.DataArray, np.ndarray],
    axis: Union[int, str] = 0,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the ensemble standard deviation.

    Parameters
    ----------
    ensemble : xarray.DataArray or numpy.ndarray
        Ensemble forecasts.
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Ensemble standard deviation.
    """
    if isinstance(ensemble, xr.DataArray):
        dim = axis
        if isinstance(axis, int):
            dim = ensemble.dims[axis]
        res = ensemble.std(dim=dim)
        return _update_history(res, "Ensemble Standard Deviation")
    return np.std(ensemble, axis=axis)


def rank_histogram(
    ensemble: Union[xr.DataArray, np.ndarray],
    obs: Union[xr.DataArray, np.ndarray],
    axis: Union[int, str] = 0,
) -> Union[xr.DataArray, np.ndarray]:
    """Count observation ranks, splitting each tie evenly across its possible ranks.

    A case contributes only when the observation and every member are finite.
    """
    labeled = isinstance(ensemble, xr.DataArray) and isinstance(obs, xr.DataArray)
    if labeled:
        dimension = ensemble.dims[axis] if isinstance(axis, int) else axis
        if dimension in obs.dims:
            raise ValueError("Observations must not include the member dimension.")
        ensemble, obs = xr.align(ensemble, obs, join="inner", copy=False)
        n = ensemble.sizes[dimension]
        lower = (ensemble < obs).sum(dimension)
        tied = (ensemble == obs).sum(dimension)
        valid = np.isfinite(ensemble).all(dimension) & np.isfinite(obs)
        lower, tied, valid = xr.broadcast(lower, tied, valid)
        lower, tied, valid = lower.data, tied.data, valid.data
    else:
        ensemble = np.asarray(ensemble)
        obs = np.expand_dims(np.asarray(obs), axis)
        n = ensemble.shape[axis]
        lower = (ensemble < obs).sum(axis)
        tied = (ensemble == obs).sum(axis)
        valid = np.isfinite(ensemble).all(axis) & np.isfinite(obs).squeeze(axis)

    # Each case adds an equal share to an interval of possible ranks.
    # Histogram the two endpoints, then take a cumulative sum.
    weights = np.where(valid, 1.0 / (tied + 1), 0.0)
    edges = np.arange(n + 3) - 0.5
    if hasattr(lower, "__dask_graph__"):
        import dask.array as da

        histogram = da.histogram
    else:
        histogram = np.histogram
    start, _ = histogram(lower, bins=edges, weights=weights)
    stop, _ = histogram(lower + tied + 1, bins=edges, weights=weights)
    result = (start - stop).cumsum(axis=0)[: n + 1]
    if labeled:
        return _update_history(xr.DataArray(result, dims="rank", coords={"rank": np.arange(n + 1)}), "Rank Histogram")
    return result


def reliability_diagram(
    obs: Union[xr.DataArray, np.ndarray],
    mod_prob: Union[xr.DataArray, np.ndarray],
    threshold: float = 0.5,
    n_bins: int = 10,
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.Dataset, Dict[str, np.ndarray]]:
    """
    Compute reliability diagram components (Aero Protocol).

    Typical Use Cases
    -----------------
    - Evaluating the calibration of probabilistic forecasts.
    - Assessing whether the forecast probabilities match the observed frequencies.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed binary outcomes (0 or 1) or continuous values (will be binarized).
    mod_prob : xarray.DataArray or numpy.ndarray
        Forecast probabilities (0 to 1).
    threshold : float, optional
        Threshold for binarizing observations if they are continuous, by default 0.5.
    n_bins : int, optional
        Number of probability bins, by default 10.
    dim : str or iterable of str, optional
        Dimension(s) along which to aggregate (xarray only).
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to aggregate (numpy only).

    Returns
    -------
    xarray.Dataset or dict
        Reliability diagram components:
        - forecast_prob: Mean forecast probability in each bin.
        - observed_freq: Observed frequency in each bin.
        - bin_counts: Number of samples in each bin.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    labeled = isinstance(obs, xr.DataArray) and isinstance(mod_prob, xr.DataArray)
    if labeled:
        obs, mod_prob = xr.align(obs, mod_prob, join="inner", copy=False)
        reduction_dim = _resolve_axis_to_dim(obs, dim if dim is not None else axis)
    else:
        obs, mod_prob = np.broadcast_arrays(obs, mod_prob)

    valid = np.isfinite(obs) & _valid_probability(mod_prob)
    event = obs >= threshold
    edges = np.linspace(0, 1, n_bins + 1)
    counts, probability_totals, events = [], [], []
    for i in range(n_bins):
        upper = mod_prob <= edges[i + 1] if i == n_bins - 1 else mod_prob < edges[i + 1]
        selected = valid & (mod_prob >= edges[i]) & upper
        if labeled:
            counts.append(selected.sum(reduction_dim))
            probability_totals.append(mod_prob.where(selected, 0).sum(reduction_dim))
            events.append((event & selected).sum(reduction_dim))
        else:
            counts.append(selected.sum(axis=axis))
            probability_totals.append(np.where(selected, mod_prob, 0).sum(axis=axis))
            events.append((event & selected).sum(axis=axis))

    if labeled:
        bins = xr.IndexVariable("bin", np.arange(n_bins))
        count = xr.concat(counts, dim=bins)
        denominator = count.where(count > 0)
        result = xr.Dataset(
            {
                "forecast_prob": xr.concat(probability_totals, dim=bins) / denominator,
                "observed_freq": xr.concat(events, dim=bins) / denominator,
                "bin_counts": count,
            }
        ).transpose(..., "bin")
        return _update_history(result, "Reliability Diagram")

    count = np.stack(counts, axis=-1)
    denominator = np.where(count > 0, count, np.nan)
    return {
        "forecast_prob": np.stack(probability_totals, axis=-1) / denominator,
        "observed_freq": np.stack(events, axis=-1) / denominator,
        "bin_counts": count,
    }
