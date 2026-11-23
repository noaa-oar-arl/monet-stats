import numpy as np
import pytest
import xarray as xr

from monet_stats.spatial_ensemble_metrics import CRPS


def test_crps_numpy_benchmarks():
    """Verify CRPS against manual analytical benchmarks."""
    # Case 1: ens=[1, 2, 3], obs=2 -> CRPS=2/9
    ens1 = np.array([[1.0, 2.0, 3.0]]).T
    obs1 = np.array([2.0])
    res1 = CRPS(ens1, obs1, axis=0)
    assert np.allclose(res1, 2 / 9)

    # Case 2: ens=[1, 5], obs=3 -> CRPS=1.0
    ens2 = np.array([[1.0, 5.0]]).T
    obs2 = np.array([3.0])
    res2 = CRPS(ens2, obs2, axis=0)
    assert np.allclose(res2, 1.0)


def test_crps_xarray_integration():
    """Verify CRPS with xarray.DataArray and dimension handling."""
    ens = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6]]).T,
        dims=["member", "site"],
        coords={"member": [1, 2, 3], "site": ["A", "B"]},
        name="forecast",
    )
    obs = xr.DataArray(np.array([2, 5]), dims=["site"], coords={"site": ["A", "B"]}, name="observation")

    # Reduction over 'member' dimension
    res = CRPS(ens, obs, axis="member")

    assert isinstance(res, xr.DataArray)
    assert res.dims == ("site",)
    assert np.allclose(res.sel(site="A"), 2 / 9)
    assert np.allclose(res.sel(site="B"), 2 / 9)
    assert "history" in res.attrs
    assert "Continuous Ranked Probability Score (CRPS)" in res.attrs["history"]


def test_crps_dask_laziness():
    """Verify CRPS remains lazy with Dask-backed arrays."""
    try:
        import dask.array as da
    except ImportError:
        pytest.skip("Dask not installed")

    ens = xr.DataArray(da.from_array(np.random.rand(10, 100), chunks=(10, 10)), dims=["member", "time"])
    obs = xr.DataArray(da.from_array(np.random.rand(100), chunks=(10,)), dims=["time"])

    res = CRPS(ens, obs, axis="member")

    # Check if the result is still a dask array
    assert hasattr(res.data, "chunks")

    # Compute and verify
    computed = res.compute()
    assert isinstance(computed.data, np.ndarray)
    assert computed.shape == (100,)


def test_crps_empty_ensemble():
    """Verify CRPS handles empty ensembles gracefully."""
    ens = np.array([]).reshape(0, 5)
    obs = np.array([1, 2, 3, 4, 5])
    res = CRPS(ens, obs, axis=0)
    assert np.all(np.isnan(res))


def test_fair_crps_and_reference_skill():
    from monet_stats import CRPSS

    ensemble = xr.DataArray([[0.0, 0.0], [2.0, 2.0]], dims=("member", "case"))
    obs = xr.DataArray([1.0, 1.0], dims="case")
    ref = xr.DataArray([[2.0, 2.0], [2.0, 2.0]], dims=("member", "case"))
    np.testing.assert_allclose(CRPS(ensemble, obs, axis="member", fair=True), [0, 0])
    assert CRPSS(ensemble, obs, ref, dim="case", fair=True, weights=[1, 2]).item() == pytest.approx(1.0)
    assert np.isnan(CRPS(ensemble.isel(member=slice(1)), obs, axis="member", fair=True)).all()


def test_spread_rmse_uses_common_support():
    from monet_stats import spread_rmse

    ens = xr.DataArray([[0.0, 0.0, 99.0], [2.0, 4.0, np.nan]], dims=("member", "case"))
    obs = xr.DataArray([0.0, 0.0, 0.0], dims="case")
    result = spread_rmse(ens, obs, dim="case")
    np.testing.assert_allclose([result.spread, result.rmse], np.sqrt([2.5, 2.5]))
    assert result.paired_count.item() == 2


def test_chunked_member_axis_ensemble_accessor_scores():
    pytest.importorskip("dask.array")
    ens = xr.DataArray([[0.0, 2.0], [2.0, 4.0]], dims=("member", "case")).chunk(member=1, case=1)
    obs = xr.DataArray([1.0, 3.0], dims="case").chunk(case=2)
    score = ens.monet_stats.crps(obs, dim="member", fair=True)
    np.testing.assert_allclose(score.compute(), 0)
    skill = ens.monet_stats.crpss(obs, obs + 1, dim="case", fair=True)
    np.testing.assert_allclose(skill.compute(), 1)
    spread = ens.monet_stats.spread_rmse(obs, dim="case")
    np.testing.assert_allclose(spread.compute().spread, 1)
