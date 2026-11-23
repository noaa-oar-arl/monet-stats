import numpy as np
import xarray as xr

from monet_stats.spatial_ensemble_metrics import reliability_diagram


def test_reliability_numpy():
    obs = np.array([0, 1, 0, 1, 0])
    mod_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    res = reliability_diagram(obs, mod_prob, n_bins=2)
    assert len(res["forecast_prob"]) == 2
    # First bin (0-0.5) has probs [0.1, 0.2, 0.3] and obs [0, 0, 0] -> freq 0
    assert np.isclose(res["observed_freq"][0], 0.0)
    # Second bin (0.5-1.0) has probs [0.9, 0.8] and obs [1, 1] -> freq 1
    assert np.isclose(res["observed_freq"][1], 1.0)


def test_reliability_xarray():
    obs = xr.DataArray([0, 1, 0, 1, 0], dims="x")
    mod_prob = xr.DataArray([0.1, 0.9, 0.2, 0.8, 0.3], dims="x")
    # Test with dim
    res = reliability_diagram(obs, mod_prob, n_bins=2, dim="x")
    assert "bin" in res.coords
    assert np.isclose(res["observed_freq"].isel(bin=0), 0.0)
    assert np.isclose(res["observed_freq"].isel(bin=1), 1.0)

    # Test without dim (all)
    res_all = reliability_diagram(obs, mod_prob, n_bins=2)
    assert "bin" in res_all.coords
    assert np.isclose(res_all["observed_freq"].isel(bin=0), 0.0)
    assert np.isclose(res_all["observed_freq"].isel(bin=1), 1.0)


if __name__ == "__main__":
    test_reliability_numpy()
    test_reliability_xarray()
    print("Reliability diagram tests passed!")


def test_probability_scores_exclude_missing_observations():
    obs = xr.DataArray([0.0, 1.0, np.nan], dims="sample")
    mod = xr.DataArray([0.4, 0.6, 0.2], dims="sample")
    result = reliability_diagram(obs, mod, n_bins=2)
    np.testing.assert_array_equal(result.bin_counts, [1, 1])
    np.testing.assert_allclose(result.observed_freq, [0, 1])


def test_chunked_reliability_uses_mergeable_bin_sums():
    import pytest

    pytest.importorskip("dask.array")
    obs = xr.DataArray([[0.0, 1.0, np.nan, 1.0], [0.0, 1.0, 0.0, 1.0]], dims=("lead", "time"))
    prob = xr.DataArray([[0.0, 1.0, 0.2, 0.7], [0.2, 0.7, 0.4, 0.5]], dims=("lead", "time"))
    desired = reliability_diagram(obs, prob, n_bins=2, dim="time")
    actual = reliability_diagram(obs.chunk(time=2), prob.chunk(time=2), n_bins=2, dim="time")
    assert all("rechunk-merge" not in layer for layer in actual.__dask_graph__().layers)
    xr.testing.assert_allclose(actual.compute(), desired)
