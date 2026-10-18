from types import SimpleNamespace

import impy as ip
import numpy as np
import pytest
from acryo import BatchLoader, Molecules, SubtomogramLoader, alignment
from acryo.tilt import no_wedge, single_axis
from dask import array as da
from numpy.testing import assert_allclose
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from cylindra import template_free
from cylindra.components.landscape import Landscape
from cylindra.components.spline import CylSpline
from cylindra.template_free import (
    AlignmentParams,
    AlignmentResult,
    AlignmentState,
    RFAAlignmentState,
    RMAAlignmentParams,
    RMAAlignmentResult,
    RMAAlignmentState,
    randomize_roll,
    resample_extracted,
    rotation_range,
)


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


def test_batch_loader_of_extracted(tmp_path):
    loader = _smooth_loader(n=8)
    ext0 = loader.replace(molecules=loader.molecules[:3])
    ext1 = loader.replace(molecules=loader.molecules[3:])
    concat = BatchLoader.from_loaders(
        [
            ext0.extract_subtomograms(tmp_path / "0"),
            ext1.extract_subtomograms(tmp_path / "1"),
        ],
        order=loader.order,
        scale=loader.scale,
        output_shape=loader.output_shape,
    )
    assert concat.molecules.count() == 8
    assert concat.output_shape == loader.output_shape
    assert_allclose(concat.asnumpy(), loader.asnumpy(), atol=1e-4)
    assert_allclose(concat.average(), loader.average(), atol=1e-4)
    fsc0, avg0 = concat.fsc_with_average(seed=0)
    fsc1, avg1 = loader.fsc_with_average(seed=0)
    assert_allclose(avg0, avg1, atol=1e-4)
    assert_allclose(fsc0["FSC-0"], fsc1["FSC-0"], atol=1e-4)


def _filament_landscape(molecules: Molecules, energy_min: float) -> Landscape:
    """Landscape of a straight filament with a funnel whose minimum is at the center."""
    zz, yy, xx = np.indices((5, 5, 5)) - 2
    energies = energy_min / (1 + zz**2 + yy**2 + xx**2)
    return Landscape(
        energies=np.stack([energies] * molecules.count()).astype(np.float32),
        molecules=molecules,
        argmax=None,
        quaternions=np.array([[0, 0, 0, 1]], dtype=np.float32),
        scale_factor=0.2,
    )


def _count_calls(monkeypatch: pytest.MonkeyPatch, obj, name: str) -> list[int]:
    count = [0]
    func = getattr(obj, name)

    def _func_and_count(*args, **kwargs):
        count[0] += 1
        return func(*args, **kwargs)

    monkeypatch.setattr(obj, name, _func_and_count)
    return count


@pytest.mark.parametrize("reverse_is_better", [False, True])
def test_rfa_step_selects_better_orientation(
    reverse_is_better: bool, monkeypatch: pytest.MonkeyPatch
):
    state = RFAAlignmentState(rng=np.random.default_rng(0))
    params = RMAAlignmentParams.init(
        cutoff=0.5,
        max_shifts=(0.4, 0.4, 0.4),
        max_rotations=(0.0, 0.0, 0.0),
        upsample_factor=1,
        temperature_time_const=0.2,
        num_trials=2,
    )
    state.results.append(RMAAlignmentResult(None, None, None, params))
    n = 6
    pos = np.stack([np.full(n, 10.0), np.arange(n) * 4.0 + 10, np.full(n, 10.0)], 1)
    mole_fw = Molecules(pos)
    mole_rv = mole_fw.rotate_by_rotvec_internal([np.pi, 0, 0])
    spl = CylSpline.line(pos[0], pos[-1])
    energy_fw, energy_rv = (-1.0, -3.0) if reverse_is_better else (-3.0, -1.0)
    forward, reverse = (
        state.prep_annealing_step(
            _filament_landscape(mole, energy), spl, (3.0, "np.inf"), 30.0
        )
        for mole, energy in [(mole_fw, energy_fw), (mole_rv, energy_rv)]
    )
    num_compute = _count_calls(monkeypatch, template_free, "compute")
    step = state.rfa_step(forward, reverse, final=True)
    # trials of both orientations are run in parallel
    assert num_compute[0] == 1
    assert step.is_reversed == reverse_is_better
    assert step.energy_forward == pytest.approx(energy_fw * n)
    assert step.energy_reverse == pytest.approx(energy_rv * n)
    mole_expected = mole_rv if reverse_is_better else mole_fw
    assert_allclose(step.molecules.pos, pos, atol=1e-6)
    assert_allclose(step.molecules.y, mole_expected.y, atol=1e-6)

    # polarity is not optimized without the reverse annealing
    forward = state.prep_annealing_step(
        _filament_landscape(mole_fw, energy_fw), spl, (3.0, "np.inf"), 30.0
    )
    step = state.rfa_step(forward, final=True)
    assert not step.is_reversed
    assert step.energy_reverse is None
    assert step.energy_forward == pytest.approx(energy_fw * n)
    assert_allclose(step.molecules.y, mole_fw.y, atol=1e-6)


@pytest.mark.parametrize(
    "max_rot, expected",
    [(0.0, (0.0, 0.0)), (20.0, (20.0, 20.0)), (30.0, (30.0, 30.0)),
     (45.0, (45.0, 22.5)), (90.0, (90.0, 30.0)), (100.0, (100.0, 25.0)),
     (180.0, (180.0, 30.0))],
)  # fmt: skip
def test_rotation_range(max_rot: float, expected: tuple[float, float]):
    max_rot_, step = rotation_range(max_rot, 30.0)
    assert (max_rot_, step) == pytest.approx(expected)
    if step > 0:
        # acryo searches int(max / step) steps in each direction
        assert int(max_rot_ / step) * step == pytest.approx(max_rot)


def test_rotation_step_of_params():
    params = RMAAlignmentParams.init(
        cutoff=0.5,
        max_shifts=(0.8, 0.8, 0.8),
        max_rotations=(0.0, 180.0, 3.0),
        upsample_factor=1,
        temperature_time_const=1.0,
        num_trials=1,
    )
    assert params.rotations == ((0.0, 0.0), (180.0, 30.0), (3.0, 3.0))
    fsc = SimpleNamespace(get_resolution=lambda _: 2.0)
    for _ in range(10):
        result = RMAAlignmentResult(fsc, None, None, params)
        params = RMAAlignmentParams.from_last_result(1.0, result, max_rot_step=30.0)
        for max_rot, step in params.rotations[1:]:
            assert 0 < step <= 30.0
            assert int(max_rot / step) * step == pytest.approx(max_rot)
    # rotations are gradually narrowed down to `min_rot_step`
    assert params.rotations[1][0] < 30.0


def _loaders_with_tilt() -> list[SubtomogramLoader]:
    loader = _smooth_loader(n=4)
    return [
        SubtomogramLoader(
            loader.image, loader.molecules, order=3, scale=loader.scale,
            output_shape=loader.output_shape, tilt_model=tilt,
        )
        for tilt in [no_wedge(), single_axis((-60, 60))]
    ]  # fmt: skip


@pytest.fixture
def num_template_rotations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count how many times template images are rotated."""
    count = [0]
    _transform = alignment.ZNCCAlignment._transform_template

    def _transform_and_count(self, *args, **kwargs):
        count[0] += 1
        return _transform(self, *args, **kwargs)

    monkeypatch.setattr(
        alignment.ZNCCAlignment, "_transform_template", _transform_and_count
    )
    return count


def test_landscape_model_is_shared(num_template_rotations: list[int]):
    loaders = _loaders_with_tilt()
    avg = ip.asarray(loaders[0].average(), axes="zyx").set_scale(zyx=0.5, unit="nm")
    params = RMAAlignmentParams.init(
        cutoff=0.5,
        max_shifts=(1.0, 1.0, 1.0),
        max_rotations=(0.0, 60.0, 0.0),
        upsample_factor=1,
        temperature_time_const=1.0,
        num_trials=1,
    )
    num_rotations = 5  # -60, -30, 0, 30, 60
    state = RMAAlignmentState(rng=np.random.default_rng(0))
    state.results.append(RMAAlignmentResult(None, avg, None, params))
    landscapes = [state.built_landscape_step(ldr, norm=False) for ldr in loaders]
    # templates are rotated only once in an iteration
    assert num_template_rotations[0] == num_rotations
    for ldr, lnd in zip(loaders, landscapes, strict=True):
        ref = Landscape.from_loader(
            ldr,
            template=avg.value,
            max_shifts=params.max_shifts,
            upsample_factor=1,
            alignment_model=alignment.ZNCCAlignment.with_params(
                rotations=params.rotations, cutoff=0.5, tilt=ldr.tilt_model
            ),
        )
        assert_allclose(lnd.energies, ref.energies, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(lnd.argmax, ref.argmax)
        assert_allclose(lnd.quaternions, ref.quaternions)
    # the tilt model of each loader is used
    assert not np.allclose(landscapes[0].energies, landscapes[1].energies)

    # templates are rotated again in the next iteration
    num_template_rotations[0] = 0
    state.results.append(RMAAlignmentResult(None, avg, None, params))
    state.built_landscape_step(loaders[0])
    assert num_template_rotations[0] == num_rotations


def test_alignment_model_is_shared(num_template_rotations: list[int]):
    loaders = _loaders_with_tilt()
    avg = ip.asarray(loaders[0].average(), axes="zyx").set_scale(zyx=0.5, unit="nm")
    params = AlignmentParams.init(0.5, (1.0, 1.0, 1.0), (0.0, 60.0, 0.0))
    state = AlignmentState(rng=np.random.default_rng(0))
    state.results.append(AlignmentResult(None, avg, None, params))
    aligned = [state.align_step(ldr) for ldr in loaders]
    assert num_template_rotations[0] == 5
    for ldr, out in zip(loaders, aligned, strict=True):
        ref = ldr.align(
            avg.value,
            max_shifts=params.max_shifts,
            rotations=params.rotations,
            cutoff=0.5,
            alignment_model=alignment.ZNCCAlignment,
        )
        assert_allclose(out.molecules.pos, ref.molecules.pos, atol=1e-5)
        assert_allclose(
            out.molecules.rotator.as_matrix(),
            ref.molecules.rotator.as_matrix(),
            atol=1e-5,
        )
        assert_allclose(out.features["score"], ref.features["score"], rtol=1e-5)


def test_landscapes_are_computed_at_once(monkeypatch: pytest.MonkeyPatch):
    loaders = _loaders_with_tilt()
    avg = ip.asarray(loaders[0].average(), axes="zyx").set_scale(zyx=0.5, unit="nm")
    params = RMAAlignmentParams.init(
        cutoff=0.5,
        max_shifts=(1.0, 1.0, 1.0),
        max_rotations=(0.0, 60.0, 0.0),
        upsample_factor=1,
        temperature_time_const=1.0,
        num_trials=1,
    )
    state = RFAAlignmentState(rng=np.random.default_rng(0))
    state.results.append(RMAAlignmentResult(None, avg, None, params))
    expected = [state.built_landscape_step(ldr, norm=False) for ldr in loaders]
    num_compute = _count_calls(monkeypatch, da, "compute")
    landscapes = state.built_landscapes_step(loaders, norm=False)
    assert num_compute[0] == 1
    for lnd, exp in zip(landscapes, expected, strict=True):
        assert_allclose(lnd.energies, exp.energies)
        np.testing.assert_array_equal(lnd.argmax, exp.argmax)


@pytest.mark.parametrize("max_num_iters", [1, 3, 8, 20])
def test_cooling_schedule(max_num_iters: int):
    state = RFAAlignmentState(max_num_iters=max_num_iters)
    final = state.final_temperature_ratio
    ratios = list[float]()
    for k in range(max_num_iters):
        # resolution keeps improving, so that FSC is not converged
        fsc = SimpleNamespace(get_resolution=lambda _, res=10.0 / (k + 1): res)
        state.results.append(SimpleNamespace(fsc=fsc))
        is_final = state.is_final_iteration()
        assert is_final == (k == max_num_iters - 1)
        ratios.append(state.temperature_ratio(final=is_final))
    # temperature is cooled down by the same ratio and reaches the final one at last
    assert ratios[-1] == final
    assert_allclose(
        ratios[:-1],
        [final ** ((k + 1) / max_num_iters) for k in range(max_num_iters - 1)],
    )
    assert all(r > final for r in ratios[:-1])


def test_default_cooling_fraction():
    # the default number of iterations corresponds to the cooling fraction of ~0.2
    assert RFAAlignmentState().cooling_fraction == pytest.approx(0.2, abs=0.05)


@pytest.mark.parametrize(
    "kwargs", [{"max_num_iters": 0}, {"final_temperature_ratio": 1.5}]
)
def test_invalid_annealing_schedule(kwargs: dict):
    with pytest.raises(ValueError):
        RFAAlignmentState(**kwargs)


def test_randomize_roll():
    mole = _smooth_loader(n=100).molecules
    out = randomize_roll(mole, np.random.default_rng(0))
    assert_allclose(out.pos, mole.pos)
    # rotated around the y-axis
    assert_allclose(out.y, mole.y, atol=1e-6)
    cos = np.sum(out.x * mole.x, axis=1)
    sin = np.sum(out.x * mole.z, axis=1)
    angles = np.arctan2(sin, cos)
    assert angles.min() < -2.5 and angles.max() > 2.5


@pytest.mark.parametrize("shift", [(0.0, 0.0, 0.0), (1.0, -2.0, 1.0)])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_landscape_rotation(shift: tuple[float, float, float], axis: int):
    rng = np.random.default_rng(0)
    img = ndi.gaussian_filter(rng.normal(size=(80, 80, 80)), 1.5).astype(np.float32)
    # general orientation, so that the local frame differs from the world frame
    rot = Rotation.from_rotvec([[0.4, -0.9, 0.7]])
    mole = Molecules(np.array([[40.0, 40.0, 40.0]]), rot)
    loader = SubtomogramLoader(img, mole, order=3, scale=1.0, output_shape=(21,) * 3)
    rotvec = np.zeros(3)
    rotvec[axis] = np.deg2rad(30)
    mole_true = mole.translate_internal([shift]).rotate_by_rotvec_internal(rotvec)
    template = loader.replace(molecules=mole_true).asnumpy()[0]
    rotations = [(0, 0)] * 3
    rotations[axis] = (30, 30)
    landscape = Landscape.from_loader(
        loader,
        template,
        max_shifts=(3.0, 3.0, 3.0),
        upsample_factor=1,
        alignment_model=alignment.ZNCCAlignment.with_params(rotations=rotations),
    )
    # integer indices of the minimum, as the results of annealing
    energies = landscape.energies[0]
    indices = np.array([np.unravel_index(np.argmin(energies), energies.shape)])
    mole_opt = landscape.transform_molecules(landscape.molecules, indices)
    assert_allclose(mole_opt.pos, mole_true.pos, atol=1e-4)
    assert_allclose(
        mole_opt.rotator.as_matrix(), mole_true.rotator.as_matrix(), atol=1e-4
    )
    # alignment of acryo loaders gives the same rotation
    aligned = loader.align(
        template,
        max_shifts=3.0,
        rotations=rotations,
        alignment_model=alignment.ZNCCAlignment,
    )
    assert_allclose(
        mole_opt.rotator.as_matrix(), aligned.molecules.rotator.as_matrix(), atol=1e-4
    )
    assert_allclose(
        mole_opt.features.select(
            "align-dzrot", "align-dyrot", "align-dxrot"
        ).to_numpy(),
        aligned.features.select("align-dzrot", "align-dyrot", "align-dxrot").to_numpy(),
        atol=1e-4,
    )


def test_update_landscape_with_rotated_molecules():
    state = RFAAlignmentState(rng=np.random.default_rng(0))
    params = RMAAlignmentParams.init(
        cutoff=0.5,
        max_shifts=(0.4, 0.4, 0.4),
        max_rotations=(0.0, 0.0, 0.0),
        upsample_factor=1,
        temperature_time_const=0.2,
        num_trials=2,
    )
    state.results.append(RMAAlignmentResult(None, None, None, params))
    n = 6
    pos = np.stack([np.full(n, 10.0), np.arange(n) * 4.0 + 10, np.full(n, 10.0)], 1)
    mole = Molecules(pos, Rotation.random(n, random_state=0))
    spl = CylSpline.line(pos[0], pos[-1])
    landscape = _filament_landscape(mole, -1.0)
    annealing = state.prep_annealing_step(landscape, spl, (3.0, "np.inf"), 30.0)
    rng = np.random.default_rng(1)
    for model in annealing.models:
        model.set_shifts(rng.integers(1, 4, size=(n, 3)).astype(np.intp))
    positions = [
        landscape.world_positions(model.shifts()) for model in annealing.models
    ]

    # landscape around the molecules at the same positions in different orientations
    mole_rot = mole.rotate_by_rotvec_internal([0.0, np.deg2rad(30), 0.0])
    new_landscape = _filament_landscape(mole_rot, -1.0)
    annealing.update_landscape(new_landscape)
    for model, pos_old in zip(annealing.models, positions, strict=True):
        pos_new = new_landscape.world_positions(model.shifts())
        # states are moved to the nearest grid points
        dist = np.linalg.norm(pos_new - pos_old, axis=1)
        assert np.all(dist <= np.sqrt(3) / 2 * new_landscape.scale_factor + 1e-5)
        # local coordinates of the model are also updated
        assert_allclose(
            model.longitudinal_distances(),
            np.linalg.norm(np.diff(pos_new, axis=0), axis=1),
            atol=1e-4,
        )

    # the next landscape is built at the same positions in the best orientations
    annealing.best_molecules = mole.rotate_by_rotvec_internal([0.0, 0.5, 0.0])
    ref = annealing.next_reference()
    assert_allclose(ref.pos, new_landscape.molecules.pos)
    assert_allclose(
        ref.rotator.as_matrix(), annealing.best_molecules.rotator.as_matrix(), atol=1e-6
    )


@pytest.mark.parametrize(
    "max_rot, expected",
    [
        (180.0, [180, 180, 180, 30, 15, 7.5, 3.75, 1.875, 1.0, 1.0]),
        (10.0, [10, 10, 10, 5, 2.5, 1.25, 1.0, 1.0, 1.0, 1.0]),
        (0.0, [0.0] * 10),
    ],
)
@pytest.mark.parametrize("cls", [AlignmentParams, RMAAlignmentParams])
def test_rotation_schedule(cls, max_rot: float, expected: list[float]):
    if cls is RMAAlignmentParams:
        params = cls.init(0.5, (1.0, 1.0, 1.0), (0.0, max_rot, 0.0), 1, 1.0, 1)
        result_cls = RMAAlignmentResult
    else:
        params = cls.init(0.5, (1.0, 1.0, 1.0), (0.0, max_rot, 0.0))
        result_cls = AlignmentResult
    fsc = SimpleNamespace(get_resolution=lambda _: 2.0)
    max_rots = list[float]()
    for _ in expected:
        max_rots.append(params.rotations[1][0])
        if max_rot > 30:
            # the step is 30 degrees until narrowed, and then 3 rotations are searched
            assert params.rotations[1][1] == pytest.approx(min(30.0, max_rots[-1]))
        result = result_cls(fsc, None, None, params)
        params = cls.from_last_result(1.0, result, min_rot_step=1.0, max_rot_step=30.0)
    assert max_rots == pytest.approx(expected)
