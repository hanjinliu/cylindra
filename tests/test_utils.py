import textwrap
from pathlib import Path

import impy as ip
import numpy as np
import polars as pl
import pytest
from acryo import Molecules
from numpy.testing import assert_allclose, assert_array_equal

from cylindra import utils, widget_utils
from cylindra.components.seam_search import (
    BooleanSeamSearcher,
    TiltScoreSeamSearcher,
    _seam_parity,
)
from cylindra.const import MoleculesHeader as Mole


def test_ints():
    assert utils.roundint(1.0) == 1
    assert utils.roundint(2.0) == 2
    assert utils.roundint(1.47) == 1
    assert utils.roundint(0.52) == 1
    assert utils.roundint(-5.3) == -5
    assert utils.roundint(-4.6) == -5
    assert utils.ceilint(5.0) == 5
    assert utils.ceilint(6.0) == 6
    assert utils.ceilint(5.01) == 6
    assert utils.ceilint(5.9) == 6


def test_make_slice_and_pad():
    assert utils.make_slice_and_pad(4, 8, 10) == (slice(4, 8), (0, 0))
    assert utils.make_slice_and_pad(-2, 5, 10) == (slice(0, 5), (2, 0))
    assert utils.make_slice_and_pad(6, 13, 10) == (slice(6, 10), (0, 3))
    assert utils.make_slice_and_pad(-5, 14, 10) == (slice(0, 10), (5, 4))
    with pytest.raises(ValueError):
        assert utils.make_slice_and_pad(-5, -2, 10) == (slice(0, 0), (5, 0))
    with pytest.raises(ValueError):
        assert utils.make_slice_and_pad(12, 16, 10) == (slice(0, 0), (0, 6))


def test_load_a_subtomogram():
    zz, yy, xx = np.indices((40, 50, 50), dtype=np.float32)
    img = ip.asarray(zz * 1e4 + yy * 1e2 + xx, axes="zyx")

    sub = utils.crop_tomogram(img, (20, 25, 25), (3, 3, 3))
    assert_allclose(
        sub[0],
        np.array(
            [
                [192424, 192425, 192426],
                [192524, 192525, 192526],
                [192624, 192625, 192626],
            ]
        ),
    )
    assert_allclose(
        sub[1],
        np.array(
            [
                [202424, 202425, 202426],
                [202524, 202525, 202526],
                [202624, 202625, 202626],
            ]
        ),
    )
    assert_allclose(
        sub[2],
        np.array(
            [
                [212424, 212425, 212426],
                [212524, 212525, 212526],
                [212624, 212625, 212626],
            ]
        ),
    )

    sub = utils.crop_tomogram(img, (1, 1, 1), (5, 5, 5))
    c = np.mean(sub)
    assert_allclose(sub[0], np.full((5, 5), c))
    assert_allclose(
        sub[1],
        np.array(
            [
                [c, c, c, c, c],
                [c, 0, 1, 2, 3],
                [c, 100, 101, 102, 103],
                [c, 200, 201, 202, 203],
                [c, 300, 301, 302, 303],
            ]
        ),
    )


def test_interval_divmod():
    def close(a, b):
        return np.isclose(a[0], b[0]) and a[1] == b[1]

    assert close(utils.interval_divmod(23, 4), (20, 5))
    assert close(utils.interval_divmod(23.7, 4), (20, 5))
    assert close(utils.interval_divmod(28.5, 5.4), (27, 5))
    assert close(utils.interval_divmod(20, 1.2), (19.2, 16))


def test_map_coordinates():
    from scipy import ndimage as ndi

    yy, xx = np.indices((100, 100))
    img = ip.asarray(np.sin(yy / 4) * np.sin(xx / 3))

    def isclose(coords):
        return assert_allclose(
            ndi.map_coordinates(img.value, coords, order=3),
            utils.map_coordinates(img, coords, order=3),
            rtol=1e-6,
            atol=1e-6,
        )

    coords = np.array([[[10, 13], [11, 16], [12, 19]], [[3, 32], [7, 26], [10, 18]]])

    isclose(coords)

    coords = np.array([[[-3, 10], [0.9, 2], [3, -6]], [[63, 28], [10, -1], [-20, -12]]])

    isclose(coords)

    coords = np.array(
        [[[-3, -6], [50, 60.2], [110, 120]], [[120, 108], [10, 40], [-20, -12]]]
    )

    isclose(coords)


def test_mirror_pcc():
    np.random.seed(1234)
    # Even number
    img = ip.random.normal(size=(128, 128))
    img_mirror = img[::-1, ::-1]
    shift1 = ip.pcc_maximum(img, img_mirror)
    shift2 = utils.mirror_pcc(img)
    assert_allclose(shift1, shift2)

    # Odd number
    img = ip.random.normal(size=(127, 127))
    img_mirror = img[::-1, ::-1]
    shift1 = ip.pcc_maximum(img, img_mirror)
    shift2 = utils.mirror_pcc(img)
    assert_allclose(shift1, shift2)
    np.random.seed()


def test_rotated_auto_zncc():
    yy, xx = np.indices((63, 63))
    img = ip.asarray(np.exp(-((yy - 24) ** 2) / 10 - (xx - 48) ** 2 / 10))

    shift = utils.rotated_auto_zncc(img, [170, 190])
    out = img.affine(translation=shift)
    assert out.argmax_nd() == (31, 31)

    shift = utils.rotated_auto_zncc(img, [150, 210])
    out = img.affine(translation=shift)
    assert out.argmax_nd() == (31, 31)


@pytest.mark.parametrize(
    "label, expected",
    [
        (np.array([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]).ravel(), 0),
        (np.array([[1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0], [0, 0, 0, 1]]).ravel(), 3),
        (np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]]).ravel(), 2),
        (np.array([[1, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1]]).ravel(), 1),
        (np.array([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0], [1, 0, 0, 1]]).ravel(), 3),
    ],
)
def test_infer_seam(label, expected: int):
    searcher = BooleanSeamSearcher(npf=4)
    assert searcher.search(label).seam_pos == expected


def test_search_seam_by_molecule_scores():
    npf, nring = 13, 6
    ring, pf = np.divmod(np.arange(npf * nring), npf)
    searcher = TiltScoreSeamSearcher(npf)
    # molecules are sorted by (nth, pf-id) if they don't have the features
    mole = Molecules(np.zeros((ring.size, 3)))
    # the template-like molecules under the seam position 5 have higher scores
    scores = np.where(_seam_parity(ring, pf, 5, npf), 1.0, -1.0)
    result = searcher._search_by_molecule_scores(mole, scores)
    assert result.seam_pos == 5
    assert_allclose(result.scores[npf:], -result.scores[:npf])
    assert_array_equal(result.get_label(ring.size), scores < 0)
    assert result.to_dataframe().columns == ["scores"]

    # shuffled molecules with the features give the same result
    perm = np.random.default_rng(0).permutation(ring.size)
    mole_feat = Molecules(
        np.zeros((ring.size, 3)),
        features=pl.DataFrame({Mole.nth: ring[perm] + 3, Mole.pf: pf[perm]}),
    )
    result_feat = searcher._search_by_molecule_scores(mole_feat, scores[perm])
    assert_allclose(result_feat.scores, result.scores)

    with pytest.raises(ValueError):  # number of molecules is different
        result.get_label(ring.size + 1)
    with pytest.raises(ValueError):  # not a multiple of npf
        searcher._search_by_molecule_scores(mole[:-1], scores[:-1])
    with pytest.raises(ValueError):  # only one ring
        searcher._search_by_molecule_scores(mole[:npf], scores[:npf])
    with pytest.raises(ValueError):  # pf-id out of range
        TiltScoreSeamSearcher(12)._search_by_molecule_scores(mole_feat, scores[perm])


def test_rust_expressions():
    assert widget_utils._polars_expr_to_str(pl.col("a") == 3) == "col('a') == 3"
    assert widget_utils._polars_expr_to_str(pl.col("f") == -4.2) == "col('f') == (-4.2)"
    assert widget_utils._polars_expr_to_str(pl.col("b") == "ts") == "col('b') == 'ts'"
    # the decimals of round are not shown in the str of polars.Expr
    with pytest.raises(ValueError):
        widget_utils._polars_expr_to_str(pl.col("a").round(2) > 1)


def test_validate():
    widget_utils._validate_expr_or_scalar(pl.col("a") == 3)
    widget_utils._validate_expr_or_scalar('pl.col("a") == 3')


def test_find_changing_point():
    idx = utils.find_changing_point([0, 1, 0, 1, 0, 1, 4, 5, 5])
    assert idx == 6


def test_read_mdoc(tmpdir):
    text = """<header>

    [T =     Tilt axis angle = 180.0, binning = 1  spot = 6  camera = 0 dosym = 0.0]

    [ZValue = 0]
    MinMaxMean = 0 10000 700
    TiltAngle = -19.993

    [ZValue = 0]
    MinMaxMean = 0 10000 700
    TiltAngle = 20.114
    """
    text = textwrap.dedent(text)
    path = Path(tmpdir) / "test.mdoc"
    path.write_text(text)
    tilt_angles = utils.read_tilt_angles_from_mdoc(path)
    assert_allclose(tilt_angles, [-19.993, 20.114])


@pytest.mark.parametrize(
    "shape_orig, shape_target",
    [
        ((7, 7, 7), (11, 11, 11)),
        ((8, 8, 8), (11, 11, 11)),
        ((7, 7, 7), (10, 10, 10)),
        ((8, 8, 8), (10, 10, 10)),
        ((7, 7, 7), (5, 5, 5)),
        ((8, 8, 8), (5, 5, 5)),
        ((7, 7, 7), (6, 6, 6)),
        ((8, 8, 8), (6, 6, 6)),
        ((7, 9, 6), (9, 8, 9)),
    ],
)
def test_fit_to_shape(shape_orig, shape_target):
    img = ip.gaussian_kernel(shape_orig)
    out = utils.fit_to_shape(img, shape_target)
    assert out.shape == shape_target
    assert out.scale == img.scale
    assert out.scale_unit == img.scale_unit
    center = tuple(s // 2 for s in shape_target)
    # due to interpolation, sometimes the center is not the maximum
    assert np.all(out[center] - out > -1e-3)
    assert np.all(np.abs(out - out[::-1]) < 1e-6)
    assert np.all(np.abs(out - out[:, ::-1]) < 1e-6)
    assert np.all(np.abs(out - out[:, :, ::-1]) < 1e-6)

def test_cluster_molecules():
    mole = Molecules(
        pos=np.zeros((9, 3)),
        features=pl.DataFrame(
            {Mole.nth: [0, 1, 2, 3, 4, 5, 1, 2, 3],
             Mole.pf: [0, 0, 0, 0, 0, 0, 1, 1, 1]}
        ),
    )
    out = utils.cluster_molecules(mole, npf=13, min_long_connections=1, min_lat_connections=2)
    assert out.shape[0] == mole.pos.shape[0]
    assert_array_equal(out, [0, 0, 0, 0, 0, 0, 0, 0, 0])

    out = utils.cluster_molecules(mole, npf=13, min_long_connections=1, min_lat_connections=4)
    assert out.shape[0] == mole.pos.shape[0]
    assert_array_equal(out, [0, 0, 0, 0, 0, 0, 1, 1, 1])


    mole = Molecules(
        pos=np.zeros((9, 3)),
        features=pl.DataFrame(
            {Mole.nth: [0, 1, 2, 3, 4, 5, 4, 5, 6],
             Mole.pf: [0, 0, 0, 0, 0, 0, 1, 1, 1]}
        ),
    )
    out = utils.cluster_molecules(mole, npf=13, min_long_connections=1, min_lat_connections=2)
    assert out.shape[0] == mole.pos.shape[0]
    assert_array_equal(out, [0, 0, 0, 0, 0, 0, 0, 0, 0])

    out = utils.cluster_molecules(mole, npf=13, min_long_connections=1, min_lat_connections=3)
    assert out.shape[0] == mole.pos.shape[0]
    assert_array_equal(out, [0, 0, 0, 0, 0, 0, 1, 1, 1])
