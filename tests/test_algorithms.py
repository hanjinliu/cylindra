import impy as ip
import numpy as np
import polars as pl
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose

from cylindra.components import CylSpline, CylTomogram
from cylindra.components.visualize import flat_view
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
    with pytest.raises(IndexError):
        tomo.fit(i=1)
    with pytest.raises(ValueError):
        tomo.fit(i=[0, 0])
    with pytest.raises(TypeError):
        tomo.fit(i="x")
    tomo.fit(i=-1)
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
    tomo.splines[0].props.loc = {"x": [0, 1, -1]}
    tomo.splines[0].props.glob = {"x": 0}
    with pytest.raises(ValueError):
        tomo.splines[0].props.glob = {"y": [0, 1]}
    tomo.splines[0].props.get_loc(pl.col("x") + 1)
    _DEFAULT = object()
    assert tomo.splines[0].props.get_loc("z", default=_DEFAULT) is _DEFAULT
    with pytest.raises(ValueError):
        tomo.splines[0].props.get_loc(pl.col("x") + 1, default=0)
    with pytest.raises(TypeError):
        tomo.splines[0].props.get_loc(1)
    assert tomo.splines[0].props.get_glob(pl.col("x") + 1) == 1
    assert tomo.splines[0].props.get_glob("z", default=11) == 11
    with pytest.raises(ValueError):
        tomo.splines[0].props.get_glob(pl.col("x") + 1, default=0)
    with pytest.raises(TypeError):
        tomo.splines[0].props.get_glob(1)


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
    assert len(tomo.splines[:]) == 2


def test_cylinder_params():
    from cylindra.components._cylinder_params import CylinderParameters

    p = CylinderParameters.solve(
        spacing=4.0,
        twist=0.02,
        rise_angle=9.0,
        radius=10.1,
        npf=13,
    )
    assert p.spacing == pytest.approx(4.0, abs=1e-6)
    assert p.rise_angle == pytest.approx(9.0, abs=1e-6)
    assert p.radius == pytest.approx(10.1, abs=1e-6)
    assert p.npf == 13
    assert p.skew > 0
    assert p.pitch == pytest.approx(4.0, abs=1e-2)
    assert p.lat_spacing == pytest.approx(5, abs=0.1)

    p = CylinderParameters.solve(
        pitch=4.0,
        skew=0.02,
        start=3,
        radius=10.5,
        npf=14,
    )
    assert p.pitch == pytest.approx(4.0, abs=1e-6)
    assert p.skew == pytest.approx(0.02, abs=1e-6)
    assert p.start == 3

    p = CylinderParameters.solve(
        spacing=4.0,
        skew=0.22,
        start=2,
        radius=10.5,
        npf=14,
    )
    assert p.spacing == pytest.approx(4.0, abs=1e-6)
    assert p.skew == pytest.approx(0.22, abs=1e-6)
    assert p.start == 2

    p = CylinderParameters.solve(
        pitch=4.0,
        skew=0.1,
        start=0,
        radius=6.6,
        npf=5,
    )

    assert p.pitch == pytest.approx(4.0, abs=1e-6)
    assert p.twist > 0
    assert p.skew == pytest.approx(0.1, abs=1e-6)
    assert p.start == 0

    p = CylinderParameters.solve(
        spacing=4.0,
        skew=0.1,
        rise_length=0.9,
        radius=6.6,
        npf=5,
    )

    assert p.spacing == pytest.approx(4.0, abs=1e-6)
    assert p.twist > 0
    assert p.skew == pytest.approx(0.1, abs=1e-6)
    assert p.rise_length == pytest.approx(0.9, abs=1e-6)

    # not enough information
    with pytest.raises(ValueError):
        CylinderParameters.solve(
            skew=0.1,
            rise_length=0.9,
            radius=6.6,
            npf=5,
        )

    with pytest.raises(ValueError):
        CylinderParameters.solve(
            pitch=4.0,
            start=1,
            radius=6.6,
            npf=5,
        )

    with pytest.raises(ValueError):
        CylinderParameters.solve(
            spacing=4.0,
            start=1,
            radius=6.6,
            npf=5,
        )

    with pytest.raises(ValueError):
        CylinderParameters.solve(
            pitch=4.0,
            twist=0.02,
            start=1,
            radius=6.6,
        )


def test_flat_view():
    spl = CylSpline().fit(coords_13pf)
    spl.props.update_glob(spacing=4.1, skew_angle=0.1, radius=10, npf=13, start=3)
    model = spl.cylinder_model()
    mole = model.to_molecules(spl).with_features(
        pl.len().mod(7).cast(pl.Float32).alias("x")
    )
    flat_view(mole, "x", spl, colors="jet")
    flat_view(mole, pl.col("x") * 2, spl, colors=lambda _: np.zeros(4))
    plt.close("all")


def test_imscale():
    from cylindra.components.imscale import ScaleOptimizer

    opt = ScaleOptimizer(0.9, 1.1)
    img_ref = ip.gaussian_kernel((11, 11, 11), sigma=2, axes="zyx")
    img = ip.gaussian_kernel((11, 11, 11), sigma=2.03, axes="zyx")
    res = opt.fit(img, img_ref)
    assert res.scale_optimal == pytest.approx(1.015, abs=1e-3)
    assert res.score_optimal > 0.95
