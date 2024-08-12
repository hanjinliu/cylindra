from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cylindra.components import CylSpline
    from cylindra.types import MoleculesLayer


def assert_molecules_equal(
    mole1: "MoleculesLayer", mole2: "MoleculesLayer", rtol=1e-5, atol=1e-2
):
    from numpy.testing import assert_allclose

    assert_allclose(mole1.molecules.pos, mole2.molecules.pos, rtol=rtol, atol=atol)
    # comparing quaternion is not safe.
    assert_allclose(mole1.molecules.x, mole2.molecules.x, rtol=rtol, atol=atol)
    assert_allclose(mole1.molecules.y, mole2.molecules.y, rtol=rtol, atol=atol)
    assert_allclose(mole1.molecules.z, mole2.molecules.z, rtol=rtol, atol=atol)


def assert_splines_close(spl0: "CylSpline", spl1: "CylSpline", tol=1e-2):
    import numpy as np

    diff = np.sqrt(np.sum((spl0.partition(n=100) - spl1.partition(n=100)) ** 2))
    assert diff < tol


TEST_DIR = Path(__file__).parent.parent.parent / "tests"
PROJECT_DIR_13PF = TEST_DIR / "test_project_13pf"
PROJECT_DIR_14PF = TEST_DIR / "test_project_14pf"
