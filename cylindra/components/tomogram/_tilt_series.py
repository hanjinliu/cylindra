from __future__ import annotations

import math
import threading
import uuid
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

import impy as ip
import numpy as np
from acryo import Molecules
from acryo.loader import LoaderBase
from dask import array as da
from numpy.typing import ArrayLike, NDArray

from cylindra.const import nm

if TYPE_CHECKING:
    from typing import Self

    from acryo import BackProjectionLoader
    from acryo.backend import Backend
    from acryo.projection import TiltSeries
    from acryo.projection._io import TiltImageSource

HalvesMethod = Literal["odd_even", "checkerboard"]
"""Method to split a tilt series into two halves of independent noise."""


class TiltSeriesSource:
    """A tilt series used as the source of a tomogram.

    The tomogram is never reconstructed as a whole. Subtomograms are reconstructed
    directly from the tilt series by ``BackProjectionLoader``, and the images of the
    tomogram (of any bin size) are dask arrays whose chunks are reconstructed on
    demand. The tilt images are preprocessed for each bin size when it is first
    needed, and cached until `release_cache` is called. The cache is enabled again if
    needed after that. The tilt series can be split into two halves of independent
    noise (see `set_halves`) to reconstruct the subtomograms of the half tomograms.

    Parameters
    ----------
    loader : BackProjectionLoader
        The base loader. Its ``tomogram_center`` must be in the coordinates of the
        tomogram (nm). The scale of the tomogram is the pixel size of the tilt series
        (``loader.tilt_series.scale``).
    shape : (int, int, int)
        Shape of the bin-1 tomogram.
    reference : ImgArray, optional
        Coarse tomogram reconstructed from the tilt series. It is only for
        visualization, and is never used for analysis.
    reference_binsize : float, optional
        Bin size of the coarse tomogram relative to the tilt images, which can be
        non-integer. Inferred from the pixel sizes by default.
    binsizes : iterable of int, default (1,)
        Bin sizes of the tomogram available for analysis.
    name : str, optional
        Name of the tomogram images.
    chunk_size : int, default 64
        Size of the chunks of the tomogram images in pixel.
    chunk_margin : int, default 16
        Each chunk is reconstructed with this margin to suppress artifacts at the
        borders between chunks.
    chunk_cache_size : int, default 1 GiB
        Maximum memory (bytes) used to cache the reconstructed chunks.
    ctf : bool, default True
        Whether the CTF of the tilt images is corrected using the CTF parameters of the
        tilt series. If false, the CTF parameters of the tilt series are ignored.
    """

    def __init__(
        self,
        loader: BackProjectionLoader,
        shape: tuple[int, int, int],
        *,
        reference: ip.ImgArray | None = None,
        reference_binsize: float | None = None,
        binsizes: Iterable[int] = (1,),
        name: str | None = None,
        chunk_size: int = 64,
        chunk_margin: int = 16,
        chunk_cache_size: int = 2**30,
        ctf: bool = True,
    ):
        if not ctf and loader.tilt_series.ctf is not None:
            loader = loader.with_tilt_series(loader.tilt_series.replace(ctf=None))
        if chunk_size <= 0 or chunk_size % 2 != 0:
            raise ValueError(f"chunk_size must be a positive even number, got {chunk_size}.")  # fmt: skip
        if chunk_margin < 0:
            raise ValueError(f"chunk_margin must be non-negative, got {chunk_margin}.")
        self._binsizes = sorted({int(b) for b in binsizes})
        if any(b < 1 for b in self._binsizes):
            raise ValueError(f"Bin sizes must be positive, got {self._binsizes}.")
        self._loader = loader
        self._ctf = bool(ctf)
        self._shape = tuple(int(s) for s in shape)
        self._reference = reference
        if reference is None:
            self._reference_binsize = None
        elif reference_binsize is None:
            self._reference_binsize = infer_reference_binsize(
                reference.scale.x, self.scale
            )
        else:
            self._reference_binsize = normalize_binsize(reference_binsize)
        self._name = name
        self._chunk_size = chunk_size
        self._chunk_margin = chunk_margin
        self._chunks = _ChunkCache(chunk_cache_size)
        self._images: dict[int, ip.LazyImgArray] = {}
        self._cache_stack: ExitStack | None = None
        self._lock = threading.Lock()
        self._halves_method: HalvesMethod | None = None
        # loaders of the half tilt series, created from the loader on demand
        self._half_loaders: tuple[BackProjectionLoader, BackProjectionLoader] | None = None  # fmt: skip
        # whether the tilt series and the reference differ from the files
        self._unsaved_tilt_series = False
        self._unsaved_reference = reference is not None and reference.source is None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(tilt_series={self.tilt_series!r}, "
            f"shape={self.shape}, cache_enabled={self.is_cache_enabled})"
        )

    @property
    def loader(self) -> BackProjectionLoader:
        """The base subtomogram loader."""
        return self._loader

    @property
    def tilt_series(self) -> TiltSeries:
        """The tilt series."""
        return self._loader.tilt_series

    @property
    def scale(self) -> nm:
        """Pixel size of the tilt images, which is the scale of the tomogram."""
        scale = self.tilt_series.scale
        assert scale is not None  # checked in BackProjectionLoader
        return scale

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the bin-1 tomogram."""
        return self._shape

    @property
    def ctf(self) -> bool:
        """Whether CTF correction is enabled.

        The CTF is corrected only if the tilt series also has the CTF parameters.
        """
        return self._ctf

    @property
    def tomogram_center(self) -> NDArray[np.float64]:
        """Origin of the tomogram space of the tilt series in nm."""
        return self._loader.tomogram_center

    @property
    def reference(self) -> ip.ImgArray | None:
        """The coarse tomogram for visualization, if given."""
        return self._reference

    @property
    def reference_binsize(self) -> int | float | None:
        """Bin size of the coarse tomogram for visualization (can be non-integer)."""
        return self._reference_binsize

    @property
    def binsizes(self) -> list[int]:
        """Bin sizes of the tomogram available for analysis."""
        return list(self._binsizes)

    @property
    def nbytes(self) -> int | None:
        """Total size of the tilt images in bytes, or None if unknown."""
        return tilt_images_nbytes(self.tilt_series)

    @property
    def has_unsaved_tilt_series(self) -> bool:
        """True if the tilt series is not the same as the file it was read from."""
        return self._unsaved_tilt_series

    @property
    def has_unsaved_reference(self) -> bool:
        """True if the reference is not the same as the file it was read from."""
        return self._unsaved_reference

    def mark_unsaved(self, tilt_series: bool = True, reference: bool = True) -> None:
        """Mark the tilt series and/or the reference as not saved in any file."""
        self._unsaved_tilt_series |= tilt_series
        self._unsaved_reference |= reference

    def set_tilt_series(self, tilt_series: TiltSeries) -> None:
        """Update the tilt series, such as the one with a refined alignment.

        If the tilt images are the same, the preprocessed tilt images are kept. The
        reconstructed chunks are released. The reference is not updated. The halves
        (see `set_halves`) are split again from the new tilt series. The CTF parameters
        of the new tilt series are ignored if the CTF is not corrected (see `ctf`).
        """
        if not self._ctf and tilt_series.ctf is not None:
            tilt_series = tilt_series.replace(ctf=None)
        with self._lock:
            self._loader = self._loader.with_tilt_series(tilt_series)
            self._chunks.clear()
            self._half_loaders = None
        self._unsaved_tilt_series = True

    def set_reference(self, reference: ip.ImgArray) -> None:
        """Update the coarse tomogram for visualization.

        The new reference must be of the same shape and bin size as the current one.
        """
        binsize = infer_reference_binsize(reference.scale.x, self.scale)
        if self._reference is not None:
            if reference.shape != self._reference.shape:
                raise ValueError(
                    f"Shape of the reference must be {self._reference.shape}, got "
                    f"{reference.shape}."
                )
            if binsize != self.reference_binsize:
                raise ValueError(
                    f"Bin size of the reference must be {self.reference_binsize}, got "
                    f"{binsize}."
                )
        self._reference = reference
        self._reference_binsize = binsize
        self._unsaved_reference = True

    def reconstruct_reference(
        self, tilt_series: TiltSeries | None = None
    ) -> ip.ImgArray:
        """Reconstruct the coarse tomogram from the tilt series.

        The tomogram is reconstructed in the same shape and bin size as the current
        reference, in the same way as `prep_reference_tomogram` (light background, in
        float16). The current tilt series is used by default.
        """
        if self._reference is None:
            raise ValueError("The tilt series has no reference tomogram.")
        binsize = self.reference_binsize
        assert binsize is not None
        if tilt_series is None:
            tilt_series = self.tilt_series
        img = reconstruct_coarse_tomogram(
            tilt_series, binsize, self._reference.shape, self._loader.tomo_hand
        )
        img.name = self._reference.name
        return img

    def project_points(self, points: NDArray[np.floating]) -> NDArray[np.float64]:
        """Project points of the tomogram (nm) to the raw tilt images.

        The points are projected by the current alignment of the tilt series, in the
        same way as the subtomograms are reconstructed (``BackProjectionLoader``).

        Parameters
        ----------
        points : (..., 3) array
            (z, y, x) coordinates of the points in the tomogram in nm.

        Returns
        -------
        (..., T, 2) array
            (y, x) positions in pixel of the raw (not binned) tilt images.
        """
        tilt_series = self.tilt_series
        projected = tilt_series.project_points(self._to_tilt_series_frame(points))
        return projected / self.scale + np.array(tilt_image_shape(tilt_series)) // 2

    def transfer_points(
        self, points: NDArray[np.floating], tilt_series: TiltSeries
    ) -> NDArray[np.float64]:
        """Transfer points of the tomogram (nm) to the tomogram of another alignment.

        An object in the tomogram is at the least-squares position of its projections
        on the tilt images by the current alignment. If the given alignment is correct,
        the projections are those of the object by the given alignment. Therefore, each
        point is transferred to the position whose projections by the given alignment
        are reconstructed at the point by the current alignment. The transfer only
        depends on the change of the alignment, not on the tilt images.

        Parameters
        ----------
        points : (N, 3) array
            (z, y, x) coordinates of the points in the tomogram in nm.
        tilt_series : TiltSeries
            Another alignment of the tilt series, such as the aligned one.

        Returns
        -------
        (N, 3) array
            (z, y, x) coordinates of the points in the tomogram of ``tilt_series``.
        """
        current = self.tilt_series
        pos = self._to_tilt_series_frame(points)
        target = current.project_points(pos)
        # The point is the least-squares position of the projections by the current
        # alignment if the residuals are orthogonal to the derivatives at the point.
        # The equations are solved by the Newton method (exact for affine alignments).
        jac = _projection_jacobian(current, pos)
        out = pos.copy()
        for _ in range(10):
            res = tilt_series.project_points(out) - target  # (N, T, 2)
            hess = np.einsum(
                "ntik,ntil->nkl", jac, _projection_jacobian(tilt_series, out)
            )
            grad = np.einsum("ntik,nti->nk", jac, res)
            step = np.linalg.solve(hess, grad[..., np.newaxis])[..., 0]
            out -= step
            if np.abs(step).max() < 1e-4:
                break
        return self._from_tilt_series_frame(out)

    def _to_tilt_series_frame(self, points: ArrayLike) -> NDArray[np.float64]:
        """Points of the tomogram (nm) relative to the center in the tilt series."""
        pos = np.array(points, dtype=np.float64) - self.tomogram_center
        if self._loader.tomo_hand == -1:
            # the tomogram is mirrored along the z axis around the center
            pos[..., 0] *= -1
        return pos

    def _from_tilt_series_frame(self, pos: ArrayLike) -> NDArray[np.float64]:
        """The inverse of `_to_tilt_series_frame`."""
        pos = np.array(pos, dtype=np.float64)
        if self._loader.tomo_hand == -1:
            pos[..., 0] *= -1
        return pos + self.tomogram_center

    def tilt_image_source(self) -> TiltImageSource:
        """The raw tilt images, which are read one by one by ``source[t]``."""
        from acryo.projection._io import TiltImageSource

        ts = self.tilt_series
        if ts.image_path is None:
            raise ValueError("Tilt images of the tilt series are not set.")
        return TiltImageSource(ts.image_path, ts.image_indices, ts.ntilts)

    @property
    def is_cache_enabled(self) -> bool:
        """True if the preprocessed tilt images are cached."""
        return self._cache_stack is not None

    def enable_cache(self) -> None:
        """Start caching the preprocessed tilt images."""
        with self._lock:
            if self._cache_stack is None:
                stack = ExitStack()
                stack.enter_context(self._loader.cache_tilt_images())
                self._cache_stack = stack

    def release_cache(self) -> None:
        """Release the preprocessed tilt images and the reconstructed chunks."""
        with self._lock:
            if self._cache_stack is not None:
                self._cache_stack.close()
                self._cache_stack = None
            self._chunks.clear()

    def get_loader(
        self,
        molecules: Molecules,
        binsize: int = 1,
        output_shape: tuple[int, int, int] | None = None,
        order: int = 1,
    ) -> BackProjectionLoader:
        """Create a loader that reconstructs subtomograms at the given bin size."""
        self.enable_cache()
        kwargs = {}
        if output_shape is not None:
            kwargs["output_shape"] = output_shape
        return self._loader.replace(
            molecules=molecules, order=order, binsize=binsize, **kwargs
        )

    @property
    def halves_method(self) -> HalvesMethod | None:
        """Method to split the tilt series into halves (see `set_halves`), or None."""
        return self._halves_method

    def set_halves(self, method: HalvesMethod | None = "odd_even") -> None:
        """Split the tilt series into two halves of independent noise.

        The subtomograms of the half tomograms are reconstructed from the halves of the
        tilt series (see `get_half_loaders`). The halves share the preprocessed tilt
        images with the whole tilt series, so they need no extra preprocessing.

        Parameters
        ----------
        method : {"odd_even", "checkerboard"} or None, default "odd_even"
            Method to split the tilt series. "odd_even" assigns the tilts to the halves
            alternately in the order of the tilt angles. "checkerboard", which splits
            the pixels of each tilt image, is not supported yet. None removes the
            halves.
        """
        check_halves_method(method)
        if method is not None and self.tilt_series.ntilts < 2:
            raise ValueError("At least 2 tilts are needed to split the tilt series.")
        with self._lock:
            self._halves_method = method
            self._half_loaders = None

    def half_loaders(self) -> tuple[BackProjectionLoader, BackProjectionLoader]:
        """The base loaders of the two halves of the tilt series."""
        with self._lock:
            if self._halves_method is None:
                raise ValueError(
                    "The tilt series is not split into halves. Call `set_halves` first."
                )
            if self._half_loaders is None:
                half0, half1 = split_tilts(
                    self.tilt_series.tilt_angles, self._halves_method
                )
                # `exclude_tilts` shares the cache of the preprocessed tilt images, and
                # the images of a half are cropped from the cached images of all tilts
                self._half_loaders = (
                    self._loader.exclude_tilts(half1),
                    self._loader.exclude_tilts(half0),
                )
            return self._half_loaders

    def get_half_loaders(
        self,
        molecules: Molecules,
        binsize: int = 1,
        output_shape: tuple[int, int, int] | None = None,
        order: int = 1,
    ) -> tuple[BackProjectionLoader, BackProjectionLoader]:
        """Create the loaders of the two halves of the tilt series (see `get_loader`)."""
        self.enable_cache()  # also enables the cache of the halves
        kwargs = {}
        if output_shape is not None:
            kwargs["output_shape"] = output_shape
        loader0, loader1 = (
            loader.replace(molecules=molecules, order=order, binsize=binsize, **kwargs)
            for loader in self.half_loaders()
        )
        return loader0, loader1

    def get_signal_noise_loaders(
        self,
        molecules: Molecules,
        binsize: int = 1,
        output_shape: tuple[int, int, int] | None = None,
        order: int = 1,
    ) -> tuple[HalfMapLoader, HalfMapLoader]:
        """Create the loaders of the sum and the difference of the half tomograms.

        The sum has the signal and the noise, and the difference has only the noise of
        the same level, as `Tomogram.get_subtomogram_and_noise_loader` of the half
        tomogram images.
        """
        loader0, loader1 = self.get_half_loaders(
            molecules, binsize=binsize, output_shape=output_shape, order=order
        )
        return HalfMapLoader.sum_and_difference(loader0, loader1)

    def shape_at(self, binsize: int) -> tuple[int, int, int]:
        """Shape of the tomogram of the given bin size."""
        return tuple(s // binsize for s in self._shape)

    def image(self, binsize: int = 1) -> ip.LazyImgArray:
        """The lazy tomogram image of the given bin size.

        Same as the binned images of a tomogram, the voxel ``i`` is at
        ``(i * binsize + (binsize - 1) / 2) * scale`` nm. Reconstructed chunks are
        cached.
        """
        if (img := self._images.get(binsize)) is None:
            img = self._images[binsize] = self._lazy_image(binsize, use_cache=True)
        return img

    def reconstruct(self, binsize: int) -> ip.ImgArray:
        """Reconstruct the whole tomogram of the given bin size."""
        return self._lazy_image(binsize, use_cache=False).compute()

    def _lazy_image(self, binsize: int, use_cache: bool) -> ip.LazyImgArray:
        if binsize < 1:
            raise ValueError(f"Bin size must be positive, got {binsize}.")
        shape = self.shape_at(binsize)
        chunks = tuple(_split_length(n, self._chunk_size) for n in shape)

        def _reconstruct(block_info=None):
            loc = block_info[None]["array-location"]
            start = tuple(a for a, _ in loc)
            size = tuple(b - a for a, b in loc)
            return self._reconstruct_chunk(binsize, start, size, use_cache)

        arr = da.map_blocks(
            _reconstruct,
            chunks=chunks,
            dtype=np.float32,
            meta=np.empty((0, 0, 0), dtype=np.float32),
            name=f"cylindra-tilt-series-bin{binsize}-{uuid.uuid4().hex}",
        )
        img = ip.LazyImgArray(arr, name=self._name, axes="zyx")
        img.set_scale(zyx=self.scale * binsize, unit="nm")
        return img

    def _reconstruct_chunk(
        self,
        binsize: int,
        start: tuple[int, int, int],
        size: tuple[int, int, int],
        use_cache: bool,
    ) -> NDArray[np.float32]:
        key = (binsize, start)
        if use_cache and (cached := self._chunks.get(key)) is not None:
            return cached.copy()  # cached chunks must not be modified
        # The cube of an odd size is reconstructed around a voxel of the chunk, so
        # that the voxels of the cube are exactly on the voxels of the tomogram.
        margin = self._chunk_margin
        center = np.array(start) + self._chunk_size // 2
        pos = (center * binsize + (binsize - 1) / 2) * self.scale
        loader = self._chunk_loader(binsize, pos[np.newaxis])
        cube = loader.construct_dask().compute(scheduler="synchronous")[0]
        out = cube[tuple(slice(margin, margin + s) for s in size)].copy()
        if use_cache:
            out.setflags(write=False)
            self._chunks.put(key, out)
            return out.copy()
        return out

    def _chunk_loader(self, binsize: int, pos: NDArray[np.float64]):
        size = self._chunk_size + 2 * self._chunk_margin + 1
        return self.get_loader(
            Molecules(pos), binsize=binsize, output_shape=(size,) * 3, order=0
        )


def check_halves_method(method: HalvesMethod | None) -> None:
    """Raise an error if the tilt series cannot be split into halves by the method."""
    if method is not None and method not in get_args(HalvesMethod):
        raise ValueError(
            f"Method must be one of {get_args(HalvesMethod)} or None, got {method!r}."
        )
    if method == "checkerboard":
        # not NotImplementedError, which is a RuntimeError ignored by thread workers
        raise ValueError(
            "Splitting the tilt images in a checkerboard pattern is not supported yet, "
            "because their preprocessed halves cannot share the cache of the "
            "preprocessed tilt images. Use 'odd_even' instead."
        )


def split_tilts(
    tilt_angles: ArrayLike, method: HalvesMethod = "odd_even"
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Indices of the tilts of the two halves of a tilt series.

    For "odd_even", the tilts are assigned to the halves alternately in the order of
    the tilt angles, so both halves sample the tilt range evenly.
    """
    if method != "odd_even":
        raise ValueError(f"Tilts cannot be split by {method!r}.")
    order = np.argsort(np.asarray(tilt_angles), kind="stable")
    return np.sort(order[0::2]), np.sort(order[1::2])


class _HalfStacks:
    """Two loaders of the halves, whose stacks of subtomograms are shared."""

    def __init__(self, loader0: LoaderBase, loader1: LoaderBase):
        self.loaders = (loader0, loader1)
        self._stacks: dict[tuple[int, ...], tuple[da.Array, da.Array]] = {}
        self._lock = threading.Lock()

    def stacks(
        self, output_shape: tuple[int, int, int], backend: Backend | None = None
    ) -> tuple[da.Array, da.Array]:
        if backend is not None:
            stack0, stack1 = (
                loader.construct_dask(output_shape, backend) for loader in self.loaders
            )
            return stack0, stack1
        key = tuple(int(s) for s in output_shape)
        with self._lock:
            if (out := self._stacks.get(key)) is None:
                stack0, stack1 = (loader.construct_dask(key) for loader in self.loaders)
                out = self._stacks[key] = (stack0, stack1)
        return out


class HalfMapLoader(LoaderBase):
    """A loader of the sum or the difference of the subtomograms of two halves.

    As the half tomograms have independent noise, the sum has the signal and the noise,
    and the difference has only the noise of the same level. The loaders created
    together by `sum_and_difference` share the subtomograms of the halves, so they are
    reconstructed only once if the loaders are computed at the same time.
    """

    def __init__(self, halves: _HalfStacks, sign: Literal[1, -1]):
        loader = halves.loaders[0]
        super().__init__(
            order=loader.order, scale=loader.scale, output_shape=loader.output_shape
        )
        self._halves = halves
        self._sign = sign

    @classmethod
    def sum_and_difference(
        cls, loader0: LoaderBase, loader1: LoaderBase
    ) -> tuple[Self, Self]:
        """Create the loaders of the sum and the difference of the two loaders."""
        halves = _HalfStacks(loader0, loader1)
        return cls(halves, 1), cls(halves, -1)

    def __repr__(self) -> str:
        kind = "sum" if self._sign > 0 else "difference"
        return (
            f"{type(self).__name__}({kind}, molecules={self.molecules!r}, "
            f"output_shape={self.output_shape}, order={self.order}, "
            f"scale={self.scale:.4f})"
        )

    @property
    def molecules(self) -> Molecules:
        return self._halves.loaders[0].molecules

    @property
    def half_loaders(self) -> tuple[LoaderBase, LoaderBase]:
        """The loaders of the two halves."""
        return self._halves.loaders

    def construct_dask(
        self,
        output_shape: int | tuple[int, ...] | None = None,
        backend: Backend | None = None,
    ) -> da.Array:
        output_shape = self._get_output_shape(output_shape)
        stack0, stack1 = self._halves.stacks(output_shape, backend)
        return stack0 + stack1 if self._sign > 0 else stack0 - stack1

    def construct_loading_tasks(
        self,
        output_shape: int | tuple[int, ...] | None = None,
        backend: Backend | None = None,
    ):
        from acryo._dask import DaskArrayList

        return DaskArrayList(list(self.construct_dask(output_shape, backend)))

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape=None,
        order: int | None = None,
        scale: float | None = None,
    ) -> Self:
        loader0, loader1 = (
            loader.replace(
                molecules=molecules, output_shape=output_shape, order=order, scale=scale
            )
            for loader in self._halves.loaders
        )
        return type(self)(_HalfStacks(loader0, loader1), self._sign)

    def binning(self, binsize: int = 2, *, compute: bool = True) -> Self:
        loader0, loader1 = (
            loader.binning(binsize, compute=compute) for loader in self._halves.loaders
        )
        return type(self)(_HalfStacks(loader0, loader1), self._sign)

    def _default_align_kwargs(self) -> dict:
        return self._halves.loaders[0]._default_align_kwargs()


def make_loader(
    tilt_series: TiltSeries,
    tomogram_center: Sequence[float],
    tomo_hand: int,
) -> BackProjectionLoader:
    """Create the loader of a tomogram reconstructed from the tilt series."""
    from acryo import BackProjectionLoader

    return BackProjectionLoader(
        tilt_series, tomogram_center=tuple(tomogram_center), tomo_hand=tomo_hand
    )


def read_tilt_series(
    path: str | Path,
    scale: nm | None = None,
    ctf: bool = True,
    local_alignment: Literal["none", "shift", "full"] = "shift",
) -> TiltSeries:
    """Read a tilt series.

    Parameters
    ----------
    path : path-like
        A JSON file of `acryo.projection.TiltSeries` (the image paths are relative to
        the file), an IMOD directive file (.edf) or a STAR file of RELION 5 tilt series.
    scale : float, optional
        Pixel size of the tilt images in nm. If not given, the pixel size in the file
        is used. Required for IMOD, which does not store the pixel size.
    ctf : bool, default True
        Whether to perform CTF correction. If false, the CTF parameters of the tilt
        series are not read (or removed from the JSON file).
    local_alignment : {"none", "shift"}, default "shift"
        Local alignment method for IMOD tilt series.
    """
    from acryo.projection import TiltSeries

    path = Path(path)
    if path.suffix == ".json":
        ts = TiltSeries.from_json(path.read_text())
        if ts.image_path is not None:
            ts = ts.replace(image_path=_resolve_image_path(ts.image_path, path.parent))
        if scale is not None:
            ts = ts.replace(scale=scale)
        if not ctf and ts.ctf is not None:
            ts = ts.replace(ctf=None)
    elif path.suffix == ".edf":
        if scale is None:
            raise ValueError("Pixel size ('scale') is required for IMOD tilt series.")
        ts = TiltSeries.from_etomo_directory(
            path.parent,
            scale=scale,
            local_alignment=local_alignment,
            ctf=ctf,
        )
    elif path.suffix == ".star":
        ts = TiltSeries.from_relion_tilt_series_star(path, scale=scale, ctf=ctf)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if ts.scale is None:
        raise ValueError(f"Pixel size of the tilt series {path} is not set.")
    return ts


def _resolve_image_path(image_path, directory: Path):
    """Resolve the path(s) of the tilt images relative to the directory."""
    if isinstance(image_path, tuple):
        return tuple(directory / p for p in image_path)
    return directory / image_path


def tilt_images_nbytes(tilt_series: TiltSeries) -> int | None:
    """Total size of the tilt images in bytes, or None if unknown."""
    from acryo.projection._io import MrcImageStack

    paths = tilt_series.image_path
    if paths is None:
        return None
    try:
        if isinstance(paths, tuple):  # a file for each tilt
            stacks = [MrcImageStack(p) for p in paths]
        else:
            stacks = [MrcImageStack(paths)] * tilt_series.ntilts
    except Exception:
        return None
    return sum(s.shape[1] * s.shape[2] * s.dtype.itemsize for s in stacks)


def tilt_image_shape(tilt_series: TiltSeries) -> tuple[int, int]:
    """(y, x) shape of the tilt images."""
    from acryo.projection._io import MrcImageStack

    paths = tilt_series.image_path
    if paths is None:
        raise ValueError("Tilt images of the tilt series are not set.")
    if isinstance(paths, tuple):  # a file for each tilt
        paths = paths[0]
    _, ny, nx = MrcImageStack(paths).shape
    return ny, nx


def coarse_tomogram_shape(
    tilt_series: TiltSeries, binsize: float, height: nm
) -> tuple[int, int, int]:
    """Shape of the coarse tomogram of the given bin size and height (nm).

    The coarse tomogram covers the tilt images in the (y, x) plane.
    """
    if tilt_series.scale is None:
        raise ValueError("Pixel size of the tilt series is not set.")
    if height <= 0:
        raise ValueError(f"Height of the tomogram must be positive, got {height}.")
    ny, nx = tilt_image_shape(tilt_series)
    nz = round(height / (tilt_series.scale * binsize))
    if abs((90 - np.mean(tilt_series.tilt_axis_angle)) % 180 - 90) > 45:
        ny, nx = nx, ny
    return max(nz, 1), max(int(ny // binsize), 1), max(int(nx // binsize), 1)


# The coarse tomogram is only for visualization, so it is reconstructed with larger
# patches, thinner blending margins and less padding than the defaults of
# `BackProjectionLoader.reconstruct_tomogram`, which is about 2x faster without
# visible artifacts.
COARSE_PATCH_SIZE = 128
COARSE_BLEND_MARGIN = 16
COARSE_PADDING = 1.5


def _reconstruct_coarse(
    tilt_series: TiltSeries,
    binsize: float,
    shape: tuple[int, int, int],
    tomo_hand: int,
    save_path: Path | None = None,
) -> NDArray[np.float16]:
    from acryo import BackProjectionLoader

    loader = BackProjectionLoader(
        tilt_series, tomo_hand=tomo_hand, padding=COARSE_PADDING
    ).binning(binsize)
    return loader.reconstruct_tomogram(
        shape,
        patch_size=COARSE_PATCH_SIZE,
        blend_margin=COARSE_BLEND_MARGIN,
        save_path=save_path,
        dtype=np.float16,
    )


def reconstruct_coarse_tomogram(
    tilt_series: TiltSeries,
    binsize: float,
    shape: tuple[int, int, int] | None,
    tomo_hand: int,
) -> ip.ImgArray:
    """Reconstruct a coarse tomogram as `TiltSeriesSource.reconstruct_reference`."""
    if shape is None:
        raise ValueError("Shape of the coarse tomogram is unknown.")
    if tilt_series.scale is None:
        raise ValueError("Pixel size of the tilt series is not set.")
    img = _reconstruct_coarse(tilt_series, binsize, tuple(shape), tomo_hand)
    img = ip.asarray(img, axes="zyx")
    return img.set_scale(zyx=tilt_series.scale * binsize, unit="nm")


def prep_reference_tomogram(
    path: str | Path,
    scale: nm | None,
    reference_scale: nm,
    tomo_hand: int,
    height: nm,
    save_coarse_tomogram: bool = True,
    ctf: bool = True,
) -> ip.ImgArray:
    """Reconstruct the coarse tomogram of the given pixel size and height (nm).

    The pixel size of the coarse tomogram does not need to be an integer multiple of
    that of the tilt series.
    """
    path = Path(path)
    ts = read_tilt_series(path, scale, ctf=ctf)
    bin_size = infer_reference_binsize(reference_scale, ts.scale)
    shape = coarse_tomogram_shape(ts, bin_size, height)
    if save_coarse_tomogram:
        save_path = path.with_name(f"{path.stem}_rec_bin{bin_size:g}.mrc")
        if save_path.exists():
            save_path.unlink()
    else:
        save_path = None
    recon = _reconstruct_coarse(ts, bin_size, shape, tomo_hand, save_path=save_path)
    if save_coarse_tomogram:
        img = ip.imread(save_path)
    else:
        img = ip.asarray(recon, axes="zyx")
    # not limited by the precision of the pixel size in the MRC header
    return img.set_scale(zyx=ts.scale * bin_size, unit="nm")


def normalize_binsize(binsize: float, tolerance: float = 1e-3) -> int | float:
    """Normalize a bin size, which can be non-integer.

    The bin size is an integer if it is close to an integer within the relative
    tolerance. Otherwise, it is rounded to 4 decimals to absorb the limited precision
    of the pixel sizes in the image headers.
    """
    binsize = float(binsize)
    if not math.isfinite(binsize) or binsize < 1 - tolerance:
        raise ValueError(f"Bin size must be 1 or larger, got {binsize}.")
    if abs(binsize / (rounded := round(binsize)) - 1) <= tolerance:
        return rounded
    return round(binsize, 4)


def infer_reference_binsize(
    reference_scale: nm, scale: nm, tolerance: float = 1e-3
) -> int | float:
    """Infer the bin size of a reference tomogram from the pixel sizes."""
    if not reference_scale / scale >= 1 - tolerance:
        raise ValueError(
            f"Pixel size of the reference tomogram ({reference_scale:.4f} nm) is "
            f"smaller than that of the tilt series ({scale:.4f} nm). Please check the "
            "pixel sizes, or specify the bin size of the reference tomogram explicitly."
        )
    return normalize_binsize(reference_scale / scale, tolerance=tolerance)


def default_tomogram_center(
    reference_shape: Sequence[int], reference_binsize: float, scale: nm
) -> tuple[float, float, float]:
    """Origin of the tomogram space of a tilt series in the tomogram coordinates."""
    offset = 0
    nz, ny, nx = (n // 2 * reference_binsize + offset for n in reference_shape)
    return (nz * scale, ny * scale, nx * scale)


def _projection_jacobian(
    tilt_series: TiltSeries, points: NDArray[np.float64], step: nm = 0.5
) -> NDArray[np.float64]:
    """(N, T, 2, 3) derivatives of the projections (y, x) by the points (z, y, x).

    The points (N, 3) are relative to the center in the tilt series, and the
    derivatives are the central differences of ``step`` nm.
    """
    offsets = np.concatenate([np.eye(3), -np.eye(3)]) * step
    projected = tilt_series.project_points(points[:, np.newaxis] + offsets)
    jac = (projected[:, :3] - projected[:, 3:]) / (2 * step)  # (N, 3, T, 2)
    return jac.transpose(0, 2, 3, 1)


def _split_length(length: int, chunk_size: int) -> tuple[int, ...]:
    nfull, rest = divmod(length, chunk_size)
    return (chunk_size,) * nfull + ((rest,) if rest else ())


class _ChunkCache:
    """Thread-safe LRU cache of arrays with a limit of the total size."""

    def __init__(self, max_bytes: int):
        self._max_bytes = max_bytes
        self._data = OrderedDict[Hashable, NDArray[np.float32]]()
        self._nbytes = 0
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> NDArray[np.float32] | None:
        with self._lock:
            if (arr := self._data.get(key)) is not None:
                self._data.move_to_end(key)
            return arr

    def put(self, key: Hashable, arr: NDArray[np.float32]) -> None:
        if arr.nbytes > self._max_bytes:
            return
        with self._lock:
            if (old := self._data.pop(key, None)) is not None:
                self._nbytes -= old.nbytes
            self._data[key] = arr
            self._nbytes += arr.nbytes
            while self._nbytes > self._max_bytes:
                _, old = self._data.popitem(last=False)
                self._nbytes -= old.nbytes

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._nbytes = 0
