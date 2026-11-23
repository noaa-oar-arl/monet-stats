import numpy as np
import pytest
import xarray as xr

from monet_stats.error_metrics import MAE
from monet_stats.uncertainty import block_bootstrap


def test_bootstrap_xarray_lazy():
    pytest.importorskip("dask.array")
    import dask.array as da

    obs_data = np.random.normal(0, 1, 100)
    mod_data = obs_data + np.random.normal(0, 0.1, 100)
    obs = xr.DataArray(da.from_array(obs_data, chunks=50), dims="time")
    mod = xr.DataArray(da.from_array(mod_data, chunks=50), dims="time")
    res = block_bootstrap(obs, mod, metric_func=MAE, n_boot=100)
    assert "mean" in res.data_vars
    assert hasattr(res["mean"].data, "chunks")
    val = res.compute()
    assert val["lower"] < val["mean"] < val["upper"]


def test_bootstrap_numpy():
    obs = np.random.normal(0, 1, 100)
    mod = obs + np.random.normal(0, 0.1, 100)
    res = block_bootstrap(obs, mod, metric_func=MAE, n_boot=100)
    assert "mean" in res
    assert res["lower"] < res["mean"] < res["upper"]


if __name__ == "__main__":
    test_bootstrap_numpy()
    try:
        test_bootstrap_xarray_lazy()
    except pytest.skip.Exception:
        pass
    print("Uncertainty metrics tests passed!")


def test_paired_bootstrap_interval_reuses_resamples():
    from monet_stats.uncertainty import paired_block_bootstrap

    obs = xr.DataArray(np.arange(8.0), dims="time")
    mod = obs + 1
    result = paired_block_bootstrap(obs, mod, mod, metric="RMSE", n_boot=30, block_size=2, seed=3)
    np.testing.assert_allclose([result.difference, result.lower, result.upper], 0)
    result = paired_block_bootstrap(obs, mod, obs + 2, metric="MAE", n_boot=30, block_size=2, seed=3)
    np.testing.assert_allclose([result.difference, result.lower, result.upper], -1)
    assert result.paired_count == 8


def test_paired_bootstrap_is_chunk_invariant_for_a_fixed_seed():
    pytest.importorskip("dask.array")
    from monet_stats.uncertainty import paired_block_bootstrap

    obs = xr.DataArray(np.arange(12.0).reshape(2, 6), dims=("lead", "time"))
    forecast = obs + xr.DataArray([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], dims="time")
    control = obs + 2
    expected = paired_block_bootstrap(obs, forecast, control, seed=1, n_boot=20, block_size=2)
    actual = forecast.chunk(time=2).monet_stats.paired_block_bootstrap(
        obs.chunk(time=3), control.chunk(time=2), seed=1, n_boot=20, block_size=2
    )
    xr.testing.assert_allclose(actual.compute(), expected)
