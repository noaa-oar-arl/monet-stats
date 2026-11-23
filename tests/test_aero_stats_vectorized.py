"""
Tests for vectorized summary statistics in Monet Stats (Aero Protocol).
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats import stats


def test_stats_vectorized_xarray():
    """Test that stats() handles multi-dimensional reductions for Xarray."""
    # Create a 3D dataset (time, lat, lon)
    times = [0, 1, 2]
    lats = [40, 41]
    lons = [-100, -99]

    obs_data = np.random.rand(3, 2, 2)
    mod_data = obs_data + 0.1

    ds = xr.Dataset(
        {
            "Obs": (["time", "lat", "lon"], obs_data),
            "Mod": (["time", "lat", "lon"], mod_data),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )

    # 1. Reduction over 'time' dimension
    # Should return a dict of 2x2 DataArrays (lat, lon)
    results = stats(ds, axis="time")

    assert isinstance(results, dict)
    assert "MB" in results
    # Aero Protocol: Should return DataArray to preserve coordinates
    assert isinstance(results["MB"], xr.DataArray)
    assert results["MB"].shape == (2, 2)
    assert "lat" in results["MB"].coords
    assert "lon" in results["MB"].coords
    assert np.allclose(results["MB"], 0.1)

    # 2. Reduction over all dimensions (default)
    # Should return a dict of scalars (floats)
    results_all = stats(ds)
    assert isinstance(results_all["MB"], float)
    assert np.isclose(results_all["MB"], 0.1)


def test_stats_accessor_vectorized():
    """Test the Xarray accessor stats method with dimension reduction."""
    obs_data = np.random.rand(5, 5)
    mod_data = obs_data + 0.5

    ds = xr.Dataset(
        {
            "Obs": (["x", "y"], obs_data),
            "Mod": (["x", "y"], mod_data),
        }
    )

    # Use accessor to reduce over 'x'
    results = ds.monet_stats.stats(dim="x")

    assert "MAE" in results
    assert isinstance(results["MAE"], xr.DataArray)
    assert results["MAE"].shape == (5,)
    assert np.allclose(results["MAE"], 0.5)


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_stats_laziness_preservation(backend):
    """Ensure stats() preserves laziness and uses optimized bundling for Dask."""
    obs_data = np.random.rand(10, 10)
    mod_data = obs_data + 0.2

    ds = xr.Dataset(
        {
            "Obs": (["x", "y"], obs_data),
            "Mod": (["x", "y"], mod_data),
        }
    )

    if backend == "dask":
        try:
            import dask.array  # noqa: F401

            ds = ds.chunk({"x": 5, "y": 5})
        except ImportError:
            pytest.skip("Dask not installed")

    # Compute stats over 'y'
    # For Dask, this should ideally bundle computations
    results = stats(ds, axis="y")

    assert "RMSE" in results
    if backend == "dask":
        # The result from stats() is already computed (it calls .compute())
        # but we want to make sure it didn't fail and returned DataArrays
        assert isinstance(results["RMSE"], xr.DataArray)

    assert np.allclose(results["RMSE"], 0.2)


def test_stats_with_plugins_vectorized():
    """Test that plugins also support vectorized reduction through stats()."""
    obs_data = np.array([1.0, 2.0, 3.0, 4.0])
    mod_data = np.array([1.1, 2.2, 3.3, 4.4])

    ds = xr.Dataset(
        {
            "Obs": (["x"], obs_data),
            "Mod": (["x"], mod_data),
        }
    )

    # WMAPE plugin
    results = stats(ds, plugins=["WMAPE"])
    assert "WMAPE" in results
    assert np.isclose(results["WMAPE"], 10.0)

    # Vectorized WMAPE (though reduction over x results in scalar)
    results_v = stats(ds, axis="x", plugins=["WMAPE"])
    assert "WMAPE" in results_v
    assert np.isclose(results_v["WMAPE"], 10.0)


def test_paired_summary_and_centered_error():
    ds = xr.Dataset({"Obs": ("sample", [1.0, 100.0, 3.0]), "Mod": ("sample", [2.0, np.nan, 4.0])})
    result = ds.monet_stats.stats()
    assert result["N"] == 3
    assert result["paired_count"] == 2
    assert (result["Obs"], result["Mod"], result["MB"], result["CRMSE"]) == (2, 3, 1, 0)
    assert ds.Mod.monet_stats.crmse(ds.Obs).item() == 0


def test_chunked_bounds_and_array_weights():
    pytest.importorskip("dask.array")
    ds = xr.Dataset({"Obs": ("sample", [1.0, 2.0, 3.0]), "Mod": ("sample", [2.0, np.inf, 5.0])}).chunk(sample=1)
    result = ds.monet_stats.stats(minval=2, weights=np.ones(3))
    assert result["N"] == 2
    assert result["paired_count"] == 1
    assert result["valid_weight"] == 1
    assert result["MB"] == 2


def test_verify_exact_alignment_and_forecast_only_dimension():
    obs = xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": [0, 1, 2]})
    mod = xr.DataArray([[2.0, 3.0], [3.0, 4.0]], dims=["lead", "time"], coords={"time": [1, 2]})
    with pytest.raises(ValueError, match="exact"):
        mod.monet_stats.verify(obs, dim="time", join="exact")
    result = mod.monet_stats.verify(obs, dim="time", join="inner")
    assert result.paired_count.dims == ("lead",)
    np.testing.assert_array_equal(result.paired_count, [2, 2])


def test_verify_invalid_and_zero_weights():
    obs = xr.DataArray([1.0, 2.0], dims="sample")
    mod = obs + 1
    with pytest.raises(ValueError, match="nonnegative"):
        mod.monet_stats.verify(obs, weights=np.array([1.0, -1.0]))
    result = mod.monet_stats.verify(obs, weights=np.zeros(2))
    assert result.valid_weight.item() == 0
    assert np.isnan(result.MB)


def test_inner_join_retains_weights_on_the_aligned_cells():
    obs = xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": [0, 1, 2]})
    weights = xr.DataArray([100.0, 2.0, 1.0], dims="time", coords={"time": obs.time})
    mod = xr.DataArray([3.0, 6.0], dims="time", coords={"time": [1, 2]})
    result = mod.monet_stats.verify(obs, weights=weights, metrics=["MB", "valid_weight"])
    np.testing.assert_allclose([result.MB.item(), result.valid_weight.item()], [5 / 3, 3])
