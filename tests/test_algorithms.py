import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from cylindra.components import CylSpline, CylTomogram
from cylindra.const import PropertyNames as H
from cylindra.project._base import MissingWedge

from ._const import TEST_DIR

coords_13pf = [[18.97, 190.0, 28.99], [18.97, 107.8, 51.48], [18.97, 35.2, 79.90]]
coords_14pf = [[21.97, 123.1, 32.98], [21.97, 83.3, 40.5], [21.97, 17.6, 64.96]]

params = [(coords_13pf, 13, 8.3, (0.0, 0.05)), (coords_14pf, 14, 7.5, (-0.25, -0.13))]


@pytest.mark.parametrize(["coords", "npf", "rise", "twist_range"], params)
def test_run_all(coords, npf, rise, twist_range):
    path = TEST_DIR / f"{npf}pf_MT.tif"
    tomo = CylTomogram.imread(path)
    repr(tomo)

    assert tomo.scale == pytest.approx(1.052, abs=1e-6)

    # the length of spline is ~80 nm
    tomo.add_spline(coords=coords)
    tomo.fit()
    tomo.refine()
    tomo.make_anchors(n=3)
    assert tomo.splines.collect_localprops() is None
    assert tomo.splines.collect_globalprops() is None
    tomo.measure_radius(positions="auto")
    tomo.measure_radius(positions=[0.2, 0.8])
    tomo.measure_radius()
    assert tomo.splines.collect_localprops() is None
    assert H.radius in tomo.splines.collect_globalprops(allow_none=False).columns

    tomo.make_anchors(interval=30)
    assert tomo.splines.collect_localprops() is None
    assert H.radius in tomo.splines.collect_globalprops(allow_none=False).columns
    assert H.spacing not in tomo.splines.collect_globalprops(allow_none=False).columns

    tomo.local_cft_params(i=0)
    assert tomo.splines.collect_localprops() is not None
    assert H.radius in tomo.splines.collect_globalprops(allow_none=False).columns
    assert H.spacing not in tomo.splines.collect_globalprops(allow_none=False).columns

    tomo.make_anchors(interval=30)
    assert tomo.splines.collect_localprops() is not None
    assert H.radius in tomo.splines.collect_globalprops(allow_none=False).columns
    assert H.spacing not in tomo.splines.collect_globalprops(allow_none=False).columns

    tomo.global_cft_params(i=0)
    assert tomo.splines.collect_localprops() is not None
    assert H.radius in tomo.splines.collect_globalprops(allow_none=False).columns
    assert H.spacing in tomo.splines.collect_globalprops(allow_none=False).columns

    spl = tomo.splines[0]
    spacing_mean = spl.props.loc[H.spacing].mean()
    spacing_glob = spl.props.get_glob(H.spacing)

    # GDP-bound microtubule has spacing in this range
    assert 4.08 < spacing_glob < 4.11
    assert spacing_glob == pytest.approx(spacing_mean, abs=5e-3)
    assert all(spl.props.loc[H.npf] == npf)
    assert all(spl.props.loc[H.rise] > rise)
    tw_min, tw_max = twist_range
    assert tw_min < spl.props.get_glob(H.twist) < tw_max

    # check cylinder parameters
    cp = tomo.splines[0].cylinder_params()
    assert cp.spacing == pytest.approx(spl.props.get_glob(H.spacing), abs=1e-6)
    assert cp.twist == pytest.approx(spl.props.get_glob(H.twist), abs=1e-6)
    assert cp.skew == pytest.approx(spl.props.get_glob(H.skew), abs=1e-6)
    assert cp.rise_angle == pytest.approx(spl.props.get_glob(H.rise), abs=1e-6)

    tomo.local_radii()
    tomo.local_cft_params(radius="local")
    tomo.local_cft_params(radius=10.2, update_glob=True)

    repr(tomo.splines[0].props)
    tomo.splines[0].props[H.spacing]
    tomo.splines[0].props.select([H.spacing])
    tomo.splines[0].props.update_glob({H.spacing: 4.0})
    tomo.splines[0].props.drop_glob(H.spacing)
    tomo.splines[0].props.drop_loc(H.spacing)
    tomo.splines[0].props.clear_loc()
    tomo.splines[0].props.clear_glob()
    del tomo.splines[0].anchors


def test_chunked_straightening():
    path = TEST_DIR / "14pf_MT.tif"
    tomo = CylTomogram.imread(path, binsize=[1, 2])

    # the length of spline is ~80 nm
    tomo.add_spline(
        np.array([[21.97, 123.1, 32.98], [21.97, 83.3, 40.5], [21.97, 17.6, 64.96]])
    )
    tomo.fit()
    tomo.make_anchors(n=3)
    tomo.measure_radius()

    st0 = tomo.straighten(i=0, chunk_length=200)
    st1 = tomo.straighten(i=0, chunk_length=32)
    assert st0.shape == st1.shape
    assert_allclose(st0.value, st1.value)
    st0 = tomo.straighten_cylindric(i=0, chunk_length=200)
    st1 = tomo.straighten_cylindric(i=0, chunk_length=32)
    assert st0.shape == st1.shape
    assert_allclose(st0.value, st1.value)

    st0 = tomo.straighten(i=0, chunk_length=200, binsize=2)
    st1 = tomo.straighten(i=0, chunk_length=32, binsize=2)
    assert st0.shape == st1.shape
    assert_allclose(st0.value, st1.value)
    st0 = tomo.straighten_cylindric(i=0, chunk_length=200, binsize=2)
    st1 = tomo.straighten_cylindric(i=0, chunk_length=32, binsize=2)
    assert st0.shape == st1.shape
    assert_allclose(st0.value, st1.value)


@pytest.mark.parametrize("orientation", [None, "PlusToMinus", "MinusToPlus"])
def test_mapping(orientation):
    path = TEST_DIR / "13pf_MT.tif"
    tomo = CylTomogram.imread(path)
    tomo.add_spline(coords=[[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])
    tomo.fit()
    tomo.splines[0].radius = 9
    tomo.splines[0].orientation = "PlusToMinus"
    tomo.global_cft_params(nsamples=2)
    tomo.map_monomers(orientation=orientation)
    tomo.map_centers(orientation=orientation)
    tomo.map_pf_line(orientation=orientation)


def test_local_cft():
    path = TEST_DIR / "13pf_MT.tif"
    tomo = CylTomogram.imread(path, binsize=[1, 2])
    tomo.add_spline(coords=[[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])
    tomo.fit()
    tomo.splines[0].radius = 9
    tomo.make_anchors(n=3)
    tomo.local_cft(i=0)
    tomo.local_cft(i=0, binsize=2)
    tomo.local_cps(i=0)


def test_global_cft():
    path = TEST_DIR / "13pf_MT.tif"
    tomo = CylTomogram.imread(path, binsize=[1, 2])
    tomo.add_spline(coords=[[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])
    tomo.fit()
    tomo.splines[0].radius = 9
    tomo.make_anchors(n=3)
    tomo.global_cft(0)
    tomo.global_cft(0, binsize=2)


@pytest.mark.parametrize(
    "tilt",
    [
        None,
        (-40, 40),
        {"kind": "none"},
        {"kind": "x", "range": (-40, 40)},
        {"kind": "y", "range": (-40, 40)},
        {"kind": "dual", "xrange": (-40, 40), "yrange": (-50, 50)},
    ],
)
def test_imread(tilt):
    path = TEST_DIR / "13pf_MT.tif"
    tomo = CylTomogram.imread(path, binsize=[1], tilt=tilt, compute=False)
    tilt0 = tomo.tilt
    wedge_model = MissingWedge.parse(tomo.tilt)
    tomo = CylTomogram.imread(
        path, binsize=[1], tilt=wedge_model.as_param(), compute=False
    )
    assert tilt0 == tomo.tilt


def test_spline_list():
    path = TEST_DIR / "13pf_MT.tif"
    tomo = CylTomogram.imread(path, binsize=[1], compute=False)
    tomo.add_spline(coords=[[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])
    tomo.add_spline(coords=[[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])
    tomo.add_spline(coords=[[18.97, 190.0, 28.99], [18.97, 187.8, 30.00]])
    for spl in tomo.splines.iter():
        assert isinstance(spl, CylSpline)
        spl.props.update_glob(length=spl.length())
    for _, spl in tomo.splines.enumerate():
        spl.make_anchors(n=3)
    splines_filt = tomo.splines.filter(pl.col("length") > 60)
    assert len(splines_filt) == 2
    assert len(tomo.splines.sort(pl.col("length"))) == 3
    for coords in tomo.splines.iter_anchor_coords():
        assert isinstance(coords, np.ndarray)
    tomo.splines.remove(tomo.splines[1])
