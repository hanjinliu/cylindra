from cylindra.components import CylTomogram
from cylindra.const import PropertyNames as H
import numpy as np
import pytest
from ._const import TEST_DIR

coords_13pf = [[18.97, 190.0, 28.99], [18.97, 107.8, 51.48], [18.97, 35.2, 79.90]]
coords_14pf = [[21.97, 123.1, 32.98], [21.97, 83.3, 40.5], [21.97, 17.6, 64.96]]

params = [(coords_13pf, 13, 8.3, (-0.1, 0.1)), (coords_14pf, 14, 7.5, (-0.5, -0.25))]


@pytest.mark.parametrize(["coords", "npf", "rise", "skew_range"], params)
def test_run_all(coords, npf, rise, skew_range):
    path = TEST_DIR / f"{npf}pf_MT.tif"
    tomo = CylTomogram.imread(path)
    repr(tomo)

    assert tomo.scale == pytest.approx(1.052, abs=1e-6)

    # the length of spline is ~80 nm
    tomo.add_spline(coords=coords)
    tomo.fit()
    tomo.refine()
    tomo.make_anchors(n=3)
    assert tomo.collect_localprops() is None
    assert tomo.collect_globalprops() is None
    tomo.set_radius()
    assert tomo.collect_localprops() is None
    assert H.radius in tomo.collect_globalprops(allow_none=False).columns

    tomo.make_anchors(interval=30)
    assert tomo.collect_localprops() is None
    assert H.radius in tomo.collect_globalprops(allow_none=False).columns
    assert H.spacing not in tomo.collect_globalprops(allow_none=False).columns

    tomo.local_ft_params(i=0)
    assert tomo.collect_localprops() is not None
    assert H.radius in tomo.collect_globalprops(allow_none=False).columns
    assert H.spacing not in tomo.collect_globalprops(allow_none=False).columns

    tomo.global_ft_params(i=0)
    assert tomo.collect_localprops() is not None
    assert H.radius in tomo.collect_globalprops(allow_none=False).columns
    assert H.spacing in tomo.collect_globalprops(allow_none=False).columns

    spl = tomo.splines[0]
    spacing_mean = spl.localprops[H.spacing].mean()
    spacing_glob = spl.props.get_glob(H.spacing)

    # GDP-bound microtubule has pitch length in this range
    assert 4.08 < spacing_glob < 4.11
    assert spacing_glob == pytest.approx(spacing_mean, abs=0.013)
    assert all(spl.localprops[H.nPF] == npf)
    assert all(spl.localprops[H.rise] > rise)
    skew_min, skew_max = skew_range
    assert skew_min < spl.props.get_glob(H.skew) < skew_max

    tomo.local_radii()
    tomo.local_ft_params(radius="local")
    tomo.local_ft_params(radius=10.2)

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
    tomo = CylTomogram.imread(path)

    # the length of spline is ~80 nm
    tomo.add_spline(
        np.array([[21.97, 123.1, 32.98], [21.97, 83.3, 40.5], [21.97, 17.6, 64.96]])
    )
    tomo.fit()
    tomo.refine()
    tomo.make_anchors(n=3)
    tomo.set_radius()

    tomo.straighten(i=0, chunk_length=200)
    tomo.straighten(i=0, chunk_length=32)
    st0 = tomo.straighten_cylindric(i=0, chunk_length=200)
    st1 = tomo.straighten_cylindric(i=0, chunk_length=32)

    from cylindra.components._localprops import polar_ft_params

    spl = tomo.splines[0]
    prop0 = polar_ft_params(st0, spl.radius)
    prop1 = polar_ft_params(st1, spl.radius)

    assert prop0.spacing == pytest.approx(prop1.spacing, abs=1e-6)
    assert prop0.skew == pytest.approx(prop1.skew, abs=1e-6)


@pytest.mark.parametrize("orientation", [None, "PlusToMinus", "MinusToPlus"])
def test_mapping(orientation):
    path = TEST_DIR / "13pf_MT.tif"
    tomo = CylTomogram.imread(path)
    tomo.add_spline(coords=[[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])
    tomo.fit()
    tomo.set_radius(radius=9)
    assert tomo.splines[0].radius == 9
    tomo.splines[0].orientation = "PlusToMinus"

    tomo.map_monomers(orientation=orientation)
    tomo.map_centers(orientation=orientation)
    tomo.map_pf_line(orientation=orientation)


def test_local_cft():
    path = TEST_DIR / "13pf_MT.tif"
    tomo = CylTomogram.imread(path, binsize=[1, 2])
    tomo.add_spline(coords=[[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])
    tomo.fit()
    tomo.set_radius(radius=9)
    tomo.make_anchors(n=3)
    tomo.local_cft(i=0)
    tomo.local_cft(i=0, binsize=2)
    tomo.local_cps(i=0)


def test_global_cft():
    path = TEST_DIR / "13pf_MT.tif"
    tomo = CylTomogram.imread(path, binsize=[1, 2])
    tomo.add_spline(coords=[[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])
    tomo.fit()
    tomo.set_radius(radius=9)
    tomo.make_anchors(n=3)
    tomo.global_cft(0)
    tomo.global_cft(0, binsize=2)
