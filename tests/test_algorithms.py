from cylindra.components import CylTomogram
from cylindra.const import PropertyNames as H
import numpy as np
import polars as pl
import pytest
from ._const import TEST_DIR

coords_13pf = [[18.97, 190.0, 28.99], [18.97, 107.8, 51.48], [18.97, 35.2, 79.90]]
coords_14pf = [[21.97, 123.1, 32.98], [21.97, 83.3, 40.5], [21.97, 17.6, 64.96]]

params = [(coords_13pf, 13, 8.3, (-0.1, 0.1)), (coords_14pf, 14, 7.5, (-0.5, -0.25))]


@pytest.mark.parametrize(["coords", "npf", "rise", "skew_range"], params)
def test_run_all(coords, npf, rise, skew_range):
    path = TEST_DIR / f"{npf}pf_MT.tif"
    tomo = CylTomogram.imread(path)

    assert abs(tomo.scale - 1.052) < 1e-6

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
    assert H.yPitch not in tomo.collect_globalprops(allow_none=False).columns

    tomo.local_ft_params(i=0)
    assert tomo.collect_localprops() is not None
    assert H.radius in tomo.collect_globalprops(allow_none=False).columns
    assert H.yPitch not in tomo.collect_globalprops(allow_none=False).columns

    tomo.global_ft_params(i=0)
    assert tomo.collect_localprops() is not None
    assert H.radius in tomo.collect_globalprops(allow_none=False).columns
    assert H.yPitch in tomo.collect_globalprops(allow_none=False).columns

    spl = tomo.splines[0]
    ypitch_mean = spl.localprops[H.yPitch].mean()
    ypitch_glob = spl.get_globalprops(H.yPitch)

    # GDP-bound microtubule has pitch length in this range
    assert 4.075 < ypitch_glob < 4.105
    assert abs(ypitch_glob - ypitch_mean) < 0.013
    assert all(spl.localprops[H.nPF] == npf)
    assert all(spl.localprops[H.riseAngle] > rise)
    skew_min, skew_max = skew_range
    assert skew_min < spl.get_globalprops(H.skewAngle) < skew_max


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

    st0 = tomo.straighten_cylindric(i=0, chunk_length=200)
    st1 = tomo.straighten_cylindric(i=0, chunk_length=32)

    from cylindra.components.cyl_tomogram import _local_dft_params_pl

    spl = tomo.splines[0]
    prop0 = pl.DataFrame(_local_dft_params_pl(st0, spl.radius))
    prop1 = pl.DataFrame(_local_dft_params_pl(st1, spl.radius))

    assert abs(prop0[H.yPitch][0] - prop1[H.yPitch][0]) < 1e-6
    assert abs(prop0[H.skewAngle][0] - prop1[H.skewAngle][0]) < 1e-6


@pytest.mark.parametrize("orientation", [None, "PlusToMinus", "MinusToPlus"])
def test_mapping(orientation):
    path = TEST_DIR / "13pf_MT.tif"
    tomo = CylTomogram.imread(path)
    tomo.add_spline(coords=[[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])
    tomo.fit()
    tomo.set_radius(radius=9)
    tomo.splines[0].radius == 9
    tomo.splines[0].orientation = "PlusToMinus"

    tomo.map_monomers(orientation=orientation)
    tomo.map_centers(orientation=orientation)
    tomo.map_pf_line(orientation=orientation)
