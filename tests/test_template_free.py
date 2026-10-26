from types import SimpleNamespace

import impy as ip
import numpy as np
import pytest
from acryo import BatchLoader, Molecules, SubtomogramLoader, alignment, pipe
from acryo.loader._extracted import SUBVOLUME_INDEX
from acryo.tilt import no_wedge, single_axis
from dask import array as da
from numpy.testing import assert_allclose
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from cylindra import template_free
from cylindra.components.landscape import Landscape
from cylindra.components.spline import CylSpline
from cylindra.template_free import (
    HALF_SET,
    AlignmentParams,
    AlignmentResult,
    AlignmentState,
    RFAAlignmentState,
    RMAAlignmentParams,
    RMAAlignmentResult,
    RMAAlignmentState,
    assign_halves,
    cutoff_of,
    fsc_weighted,
    randomize_roll,
    resample_extracted,
    rotation_range,
    split_into_halves,
)
from cylindra.widget_utils import FscResult


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
    for _ in range(10):
        params = params.next_iteration(1.0, 0.5, max_rot_step=30.0)
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


def _decaying_fsc(size: int = 20, scale: float = 0.5) -> FscResult:
    """FSC of a box of `size` that decays from 1 to 0 around 0.25 (1/pixel)."""
    dfreq = 1.5 / size
    freq = (np.arange(int(np.sqrt(3) / 2 / dfreq)) + 0.5) * dfreq
    fsc = 1 / (1 + (freq / 0.25) ** 4)
    return FscResult(freq, fsc, np.zeros_like(fsc), scale)


def test_landscape_of_half_set(num_template_rotations: list[int]):
    loader = _loaders_with_tilt()[1]
    half_avgs = tuple(
        ip.asarray(loader.replace(molecules=loader.molecules[sl]).average(), axes="zyx")
        .set_scale(zyx=0.5, unit="nm")
        for sl in [slice(0, 2), slice(2, 4)]
    )  # fmt: skip
    params = RMAAlignmentParams.init(
        cutoff=0.5,
        max_shifts=(1.0, 1.0, 1.0),
        max_rotations=(0.0, 60.0, 0.0),
        upsample_factor=1,
        temperature_time_const=1.0,
        num_trials=1,
    )
    fsc = _decaying_fsc()
    state = RMAAlignmentState(rng=np.random.default_rng(0))
    state.results.append(RMAAlignmentResult(fsc, None, None, params, half_avgs))
    halves = [0, 1, 0, 1]
    landscapes = [
        state.built_landscape_step(loader, norm=False, half=half) for half in halves
    ]
    # templates of each half set are rotated only once in an iteration
    assert num_template_rotations[0] == 5 * 2
    for half, lnd in zip(halves, landscapes, strict=True):
        # the templates are weighted by the FSC between the half sets
        ref = Landscape.from_loader(
            loader,
            template=fsc_weighted(half_avgs[half].value, fsc),
            max_shifts=params.max_shifts,
            upsample_factor=1,
            alignment_model=alignment.ZNCCAlignment.with_params(
                rotations=params.rotations, cutoff=0.5, tilt=loader.tilt_model
            ),
        )
        assert_allclose(lnd.energies, ref.energies, rtol=1e-5, atol=1e-6)
    assert not np.allclose(landscapes[0].energies, landscapes[1].energies)


@pytest.mark.parametrize("mask_shape", [None, (20, 20, 20), (16, 16, 16)])
def test_gold_standard_fsc(mask_shape: tuple[int, int, int] | None):
    loader = _smooth_loader(n=2)
    if mask_shape is None:
        mask = None
    else:
        center = (np.array(mask_shape) - 1) / 2
        dist = np.linalg.norm(
            np.indices(mask_shape) - center[:, None, None, None], axis=0
        )
        mask = (dist < mask_shape[0] / 3).astype(np.float32)
    halves = tuple(
        loader.replace(molecules=loader.molecules[i : i + 1]) for i in range(2)
    )
    fsc, avg, half_avgs, mask_used = template_free._fsc_with_average(
        halves, mask, np.random.default_rng(0)
    )
    if mask is None:
        assert mask_used is None
    else:
        expected_mask = loader.normalize_mask(mask, reshape_to=(20, 20, 20))
        assert_allclose(mask_used, expected_mask)
    # same as randomly splitting two molecules, as FSC is symmetric
    fsc_ref, avg_ref = loader.fsc_with_average(mask, seed=0)
    assert_allclose(fsc.freq, fsc_ref["freq"])
    assert_allclose(fsc.mean, fsc_ref["FSC-0"], atol=1e-4)
    assert_allclose(avg.value, avg_ref, atol=1e-5)
    assert fsc.scale == loader.scale
    for half_avg, half in zip(half_avgs, halves, strict=True):
        assert half_avg.scale.x == pytest.approx(loader.scale)
        expected = half.average()
        assert_allclose(
            half_avg.value - half_avg.value.mean(),
            expected - expected.mean(),
            atol=1e-5,
        )


def test_gold_standard_average_is_weighted():
    loader = _smooth_loader(n=4)
    halves = (
        loader.replace(molecules=loader.molecules[:1]),
        loader.replace(molecules=loader.molecules[1:]),
    )
    state = RMAAlignmentState(rng=np.random.default_rng(0))
    result = state.fsc_step_init(halves, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 1, 1.0, 1)
    assert result.half_avgs is not None
    # average of all the molecules, not the mean of the half sets
    expected = loader.average()
    assert_allclose(
        result.avg.value - result.avg.value.mean(),
        expected - expected.mean(),
        atol=1e-5,
    )
    # the half sets are not used without splitting
    state = RMAAlignmentState(rng=np.random.default_rng(0))
    result = state.fsc_step_init(loader, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 1, 1.0, 1)
    assert result.half_avgs is None
    with pytest.raises(ValueError):
        result.template(0)


@pytest.mark.parametrize("halved", [False, True])
@pytest.mark.parametrize("cls", [AlignmentState, RMAAlignmentState])
def test_first_iteration_is_masked(cls, halved: bool):
    loader = _smooth_loader(n=4)
    inputs = assign_halves(loader, np.random.default_rng(1)) if halved else loader
    mask = pipe.soft_otsu(sigma=1.0, radius=1.0)
    state = cls(rng=np.random.default_rng(0), mask=mask)
    if cls is RMAAlignmentState:
        args = (1, 1.0, 1)
    else:
        args = ()
    result = state.fsc_step_init(inputs, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), *args)
    # the mask of the first alignment is created from the initial average (before
    # normalization, so that the thresholding may differ by rounding errors)
    expected_mask = mask.convert(result.avg.value, loader.scale)
    assert result.mask is not None
    assert np.mean(np.abs(result.mask - expected_mask)) < 0.01
    # FSC is calculated with the mask, so that it is comparable to the later ones
    fsc_masked, *_ = template_free._fsc_with_average(
        inputs, result.mask, np.random.default_rng(0)
    )
    fsc_unmasked, *_ = template_free._fsc_with_average(
        inputs, None, np.random.default_rng(0)
    )
    assert_allclose(result.fsc.mean, fsc_masked.mean, atol=1e-6)
    assert not np.allclose(result.fsc.mean, fsc_unmasked.mean, atol=1e-3)


def _fake_fsc(resolution: float) -> SimpleNamespace:
    return SimpleNamespace(get_resolution=lambda _: resolution)


@pytest.mark.parametrize("keep_best_template", [False, True])
def test_template_result(keep_best_template: bool):
    state = RMAAlignmentState(keep_best_template=keep_best_template)
    # 0.0 means FSC below the threshold at all frequencies, and NaN means above
    resolutions = [3.0, 4.0, 3.5, 2.5, 2.5, 0.0, np.nan]
    expected = [0, 0, 0, 3, 3, 3, 6]
    for i, res in enumerate(resolutions):
        state.results.append(SimpleNamespace(fsc=_fake_fsc(res)))
        idx = expected[i] if keep_best_template else i
        assert state.template_result is state.results[idx]


def _rma_params(cutoff: float, max_rot: float) -> RMAAlignmentParams:
    return RMAAlignmentParams.init(
        cutoff=cutoff,
        max_shifts=(1.0, 1.0, 1.0),
        max_rotations=(0.0, max_rot, 0.0),
        upsample_factor=1,
        temperature_time_const=1.0,
        num_trials=1,
    )


def test_landscape_of_best_template():
    loader = _loaders_with_tilt()[1]
    avg_good, avg_bad = (
        ip.asarray(loader.replace(molecules=loader.molecules[sl]).average(), axes="zyx")
        .set_scale(zyx=0.5, unit="nm")
        for sl in [slice(0, 2), slice(2, 4)]
    )  # fmt: skip
    state = RMAAlignmentState(rng=np.random.default_rng(0), keep_best_template=True)
    params = _rma_params(0.3, 30.0)
    state.results.append(RMAAlignmentResult(_fake_fsc(2.0), avg_good, None, params))
    state.results.append(
        RMAAlignmentResult(_fake_fsc(3.0), avg_bad, None, _rma_params(0.5, 60.0))
    )
    landscape = state.built_landscape_step(loader, norm=False)
    # template of the best FSC, with the alignment parameters of the last result
    ref = Landscape.from_loader(
        loader,
        template=avg_good.value,
        max_shifts=params.max_shifts,
        upsample_factor=1,
        alignment_model=alignment.ZNCCAlignment.with_params(
            rotations=state.results[-1].params.rotations,
            cutoff=0.5,
            tilt=loader.tilt_model,
        ),
    )
    assert_allclose(landscape.energies, ref.energies, rtol=1e-5, atol=1e-6)
    assert_allclose(landscape.quaternions, ref.quaternions)


def test_mask_of_best_template():
    loader = _smooth_loader(n=4)
    converted = list[np.ndarray]()

    def _record(img, scale):
        converted.append(img)
        return np.ones_like(img)

    state = RMAAlignmentState(
        rng=np.random.default_rng(0),
        mask=pipe.ImageConverter(_record),
        keep_best_template=True,
    )
    avgs = [
        ip.asarray(np.full(loader.output_shape, val, dtype=np.float32), axes="zyx")
        for val in [1.0, 2.0]
    ]
    for res, avg in zip([2.0, 3.0], avgs, strict=True):
        state.results.append(
            RMAAlignmentResult(_fake_fsc(res), avg, None, _rma_params(0.5, 0.0))
        )
    result = state.fsc_step(loader)
    # mask is created from the template, not from the last average
    assert len(converted) == 1
    assert_allclose(converted[0], avgs[0].value)
    assert_allclose(result.mask, 1.0)


@pytest.mark.parametrize("kind", ["tomogram", "extracted", "batch"])
@pytest.mark.parametrize("n", [7, 8])
def test_assign_halves(tmp_path, kind: str, n: int):
    loader = _smooth_loader(n=n)
    if kind == "extracted":
        # extracted subvolumes, as in template-free alignment of a tomogram
        loader = loader.extract_subtomograms(tmp_path / "ext")
    elif kind == "batch":
        # loader of the extracted subvolumes, as in batch alignment
        loader = BatchLoader.from_loaders(
            [loader.extract_subtomograms(tmp_path / "ext")],
            order=loader.order,
            scale=loader.scale,
            output_shape=loader.output_shape,
        )
    assigned = assign_halves(loader, np.random.default_rng(0))
    labels = assigned.molecules.features[HALF_SET].to_numpy()
    # molecules are randomly split into half sets of the same size
    assert sorted(np.bincount(labels)) == [n // 2, n - n // 2]
    assert_allclose(assigned.molecules.pos, loader.molecules.pos)
    assert any(
        assign_halves(loader, np.random.default_rng(seed)).molecules.features[HALF_SET].to_list() != labels.tolist()
        for seed in range(1, 5)
    )  # fmt: skip
    # FSC and the averages are the same as those of the loaders of the half sets
    halves = tuple(loader.filter(labels == half) for half in (0, 1))
    fsc, avg, half_avgs, _ = template_free._fsc_with_average(
        assigned, None, np.random.default_rng(0)
    )
    fsc_ref, avg_ref, half_avgs_ref, _ = template_free._fsc_with_average(
        halves, None, np.random.default_rng(0)
    )
    assert_allclose(fsc.mean, fsc_ref.mean, atol=1e-6)
    assert_allclose(avg.value, avg_ref.value, atol=1e-5)
    for half_avg, half_avg_ref in zip(half_avgs, half_avgs_ref, strict=True):
        assert_allclose(half_avg.value, half_avg_ref.value, atol=1e-5)
    # the half sets are temporary
    out = template_free.as_alignment_output(loader.molecules, [assigned.molecules])
    assert (
        out.features.columns
        == loader.molecules.features.drop(SUBVOLUME_INDEX, strict=False).columns
    )
    with pytest.raises(ValueError):
        assign_halves(loader.head(1), np.random.default_rng(0))


def test_gold_standard_alignment():
    loader = assign_halves(_smooth_loader(n=6), np.random.default_rng(0))
    labels = loader.molecules.features[HALF_SET].to_numpy()
    # the half sets are not contiguous, so that the order must be restored
    assert np.any(np.diff(labels) < 0)
    mask = pipe.soft_otsu(sigma=1.0, radius=1.0)
    state = AlignmentState(rng=np.random.default_rng(0), mask=mask)
    result = state.fsc_step_init(loader, (1.0, 1.0, 1.0), (0.0, 30.0, 0.0))
    assert result.half_avgs is not None
    assert result.params.cutoff == pytest.approx(
        cutoff_of(result.fsc, loader.scale, 20, weighted=True)
    )
    aligned = state.align_step(loader)
    assert aligned.molecules.features[HALF_SET].to_list() == labels.tolist()
    # molecules of each half set are aligned to the average of the same half set,
    # weighted by the FSC between the half sets, using the mask of the average
    for half in (0, 1):
        rows = np.flatnonzero(labels == half)
        ref = loader.filter(labels == half).align(
            fsc_weighted(result.half_avgs[half].value, result.fsc),
            mask=result.mask,
            max_shifts=result.params.max_shifts,
            rotations=result.params.rotations,
            cutoff=result.params.cutoff,
            alignment_model=alignment.ZNCCAlignment,
        )
        out = aligned.molecules[rows]
        assert_allclose(out.pos, ref.molecules.pos, atol=1e-5)
        assert_allclose(out.features["score"], ref.features["score"], rtol=1e-5)
    result = state.fsc_step(aligned)
    assert result.half_avgs is not None
    assert result.params.num_iter == 1


@pytest.mark.parametrize(
    "value, factor", [(1.0, 1.0), (0.5, np.sqrt(2 / 3)), (-0.2, 0.0)]
)
def test_fsc_weighted_by_constant(value: float, factor: float):
    img = np.random.default_rng(0).normal(size=(20, 18, 17)).astype(np.float32)
    # FSC of the same value at all frequencies, weighted by sqrt(2 FSC / (1 + FSC))
    freq = np.linspace(0.0, 1.0, 21)
    fsc = FscResult(freq, np.full(21, value), np.zeros(21), 0.5)
    out = fsc_weighted(img, fsc)
    assert out.dtype == np.float32
    assert_allclose(out, img * factor, atol=1e-5)


def test_fsc_weighted_by_frequency():
    img = np.random.default_rng(0).normal(size=(20, 20, 20)).astype(np.float32)
    # FSC of 1 for the first 3 shells and 0 for the others
    freq = (np.arange(12) + 0.5) * 0.075
    fsc = FscResult(freq, np.where(freq < 0.2, 1.0, 0.0), np.zeros(12), 0.5)
    ft_in = np.fft.rfftn(img)
    ft_out = np.fft.rfftn(fsc_weighted(img, fsc))
    grids = np.meshgrid(
        np.fft.fftfreq(20), np.fft.fftfreq(20), np.fft.rfftfreq(20), indexing="ij"
    )
    radius = np.sqrt(sum(grid**2 for grid in grids))
    # weight is interpolated between the shells
    low, high = radius <= freq[2], radius >= freq[3]
    assert_allclose(ft_out[low], ft_in[low], atol=1e-2)
    assert_allclose(ft_out[high], 0.0, atol=1e-2)
    mid = ~(low | high)
    assert np.all(np.abs(ft_out[mid]) < np.abs(ft_in[mid]))


@pytest.mark.parametrize("size", [20, 64])
@pytest.mark.parametrize("weighted", [False, True])
def test_cutoff_of(size: int, weighted: bool):
    freq = (np.arange(12) + 0.5) * 0.075
    # FSC decreases linearly, crossing 0.5 at 0.2 and 0.143 at 0.3428 (1/pixel)
    fsc = FscResult(freq, 1 - freq / 0.4, np.zeros(12), 0.5)
    freq_0500, freq_0143 = 0.2, (1 - 0.143) * 0.4
    if weighted:
        # the larger of 10 Fourier pixels beyond FSC=0.5 and 5 beyond FSC=0.143
        expected = max(freq_0500 + 10 / size, freq_0143 + 5 / size)
    else:
        expected = freq_0143
    assert cutoff_of(fsc, 0.5, size, weighted) == pytest.approx(expected)
    if weighted:
        # FSC above the thresholds at all frequencies, or below them
        fsc_high = FscResult(freq, np.ones(12), np.zeros(12), 0.5)
        fsc_low = FscResult(freq, np.zeros(12), np.zeros(12), 0.5)
        assert cutoff_of(fsc_high, 0.5, size, weighted) == 1.0
        assert cutoff_of(fsc_low, 0.5, size, weighted) == pytest.approx(10 / size)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_into_halves(seed: int):
    counts = [40, 10, 25, 25, 5, 30]
    halves = split_into_halves(counts, np.random.default_rng(seed))
    assert sorted(set(halves)) == [0, 1]
    totals = [
        sum(c for c, h in zip(counts, halves, strict=True) if h == i) for i in (0, 1)
    ]
    assert abs(totals[0] - totals[1]) <= max(counts)
    assert split_into_halves([3, 3], np.random.default_rng(seed)) in ([0, 1], [1, 0])
    with pytest.raises(ValueError):
        split_into_halves([10], np.random.default_rng(seed))


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
    else:
        params = cls.init(0.5, (1.0, 1.0, 1.0), (0.0, max_rot, 0.0))
    max_rots = list[float]()
    for _ in expected:
        max_rots.append(params.rotations[1][0])
        if max_rot > 30:
            # the step is 30 degrees until narrowed, and then 3 rotations are searched
            assert params.rotations[1][1] == pytest.approx(min(30.0, max_rots[-1]))
        params = params.next_iteration(1.0, 0.5, min_rot_step=1.0, max_rot_step=30.0)
        assert isinstance(params, cls)
    assert max_rots == pytest.approx(expected)


@pytest.mark.parametrize("cls", [AlignmentState, RMAAlignmentState])
@pytest.mark.parametrize("keep_best_template", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_cutoff_of_template(
    monkeypatch: pytest.MonkeyPatch, cls, keep_best_template: bool, weighted: bool
):
    resolutions = iter([2.0, 4.0, 1.0])  # initial, worse and better
    avg = ip.asarray(np.zeros((64, 64, 64), dtype=np.float32), axes="zyx")
    # templates are weighted by FSC if the half sets are aligned independently
    half_avgs = (avg, avg) if weighted else None
    monkeypatch.setattr(
        template_free,
        "_fsc_with_average",
        lambda *_: (_fake_fsc(next(resolutions)), avg, half_avgs, None),
    )
    loader = SimpleNamespace(scale=0.5)
    state = cls(keep_best_template=keep_best_template)
    if cls is RMAAlignmentState:
        state.fsc_step_init(loader, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 1, 1.0, 1)
    else:
        state.fsc_step_init(loader, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    state.fsc_step(loader)
    state.fsc_step(loader)
    # cutoff of the parameters of each result is determined by the FSC of the template
    # of the next alignment, which is the result itself unless the template is kept
    expected = [0.5 / 2.0, 0.5 / (2.0 if keep_best_template else 4.0), 0.5 / 1.0]
    if weighted:
        # 10 Fourier pixels beyond the resolution, as the fake FSC crosses all the
        # thresholds at the same frequency
        expected = [c + 10 / 64 for c in expected]
    assert [r.params.cutoff for r in state.results] == pytest.approx(expected)
    assert [r.params.num_iter for r in state.results] == [0, 1, 2]


def test_params_format():
    params = _rma_params(0.5, 0.0)
    # labeled by the iteration in which the parameters are used
    assert params.format().startswith("Iteration 1:")
    assert "Upsample factor" in params.format()
    params = params.next_iteration(1.0, 0.25)
    assert params.format().startswith("Iteration 2:")
    assert "Relative cutoff frequency = 0.250" in params.format()
    assert "Upsample factor" not in params.format()
    params = AlignmentParams.init(0.5, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    assert params.format().startswith("Iteration 1:")
    assert params.next_iteration(1.0, 0.25).format().startswith("Iteration 2:")
