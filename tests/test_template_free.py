import numpy as np
import pytest
from acryo import Molecules, SubtomogramLoader
from acryo.loader import ConcatenatedLoader
from numpy.testing import assert_allclose
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from cylindra.template_free import resample_extracted


def _smooth_loader(n: int = 6) -> SubtomogramLoader:
    rng = np.random.default_rng(0)
    img = ndi.gaussian_filter(rng.normal(size=(64, 64, 64)), 2).astype(np.float32)
    pos = rng.uniform(12, 20, size=(n, 3))
    mole = Molecules(pos, Rotation.random(n, random_state=1))
    return SubtomogramLoader(img, mole, order=3, scale=0.5, output_shape=(20, 20, 20))


@pytest.mark.parametrize("prefilter", [False, True])
@pytest.mark.parametrize("rotate", [False, True])
def test_resample_extracted(tmp_path, prefilter: bool, rotate: bool):
    loader = _smooth_loader()
    ext = loader.extract_subtomograms(tmp_path / "ext", apply_prefilter=prefilter)
    rng = np.random.default_rng(2)
    n = ext.molecules.count()
    mole = ext.molecules.translate_internal(rng.uniform(-1.2, 1.2, size=(n, 3)))
    if rotate:
        rotvec = rng.normal(scale=0.1, size=(n, 3))
        mole = mole.rotate_by(Rotation.from_rotvec(rotvec))
    resampled = resample_extracted(ext, mole).asnumpy()
    direct = loader.replace(molecules=mole).asnumpy()
    # voxels near the edges are filled with the mean value after resampling
    sl = (slice(None),) + (slice(5, 15),) * 3
    assert_allclose(resampled[sl], direct[sl], atol=2e-3)


def test_concatenated_loader(tmp_path):
    loader = _smooth_loader(n=8)
    ext0 = loader.replace(molecules=loader.molecules[:3])
    ext1 = loader.replace(molecules=loader.molecules[3:])
    concat = ConcatenatedLoader(
        [
            ext0.extract_subtomograms(tmp_path / "0"),
            ext1.extract_subtomograms(tmp_path / "1"),
        ]
    )
    assert concat.molecules.count() == 8
    assert concat.output_shape == loader.output_shape
    assert_allclose(concat.asnumpy(), loader.asnumpy(), atol=1e-4)
    assert_allclose(concat.average(), loader.average(), atol=1e-4)
    fsc0, avg0 = concat.fsc_with_average(seed=0)
    fsc1, avg1 = loader.fsc_with_average(seed=0)
    assert_allclose(avg0, avg1, atol=1e-4)
    assert_allclose(fsc0["FSC-0"], fsc1["FSC-0"], atol=1e-4)
