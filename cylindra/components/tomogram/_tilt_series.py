from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import impy as ip
import numpy as np
from acryo import Molecules
from dask import array as da
from numpy.typing import NDArray

from cylindra.const import nm

if TYPE_CHECKING:
    from acryo import BackProjectionLoader
    from acryo.projection import TiltSeries


class TiltSeriesSource:
    """A tilt series used as the source of a tomogram.

    The tomogram is never reconstructed as a whole. Subtomograms are reconstructed
    directly from the tilt series by ``BackProjectionLoader``, and the images of the
    tomogram (of any bin size) are dask arrays whose chunks are reconstructed on
    demand. The tilt images are preprocessed for each bin size when it is first
    needed, and cached until `release_cache` is called. The cache is enabled again if
    needed after that.

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
    """

    def __init__(
        self,
        loader: BackProjectionLoader,
        shape: tuple[int, int, int],
        *,
        reference: ip.ImgArray | None = None,
        binsizes: Iterable[int] = (1,),
        name: str | None = None,
        chunk_size: int = 64,
        chunk_margin: int = 16,
        chunk_cache_size: int = 2**30,
    ):
        if chunk_size <= 0 or chunk_size % 2 != 0:
            raise ValueError(f"chunk_size must be a positive even number, got {chunk_size}.")  # fmt: skip
        if chunk_margin < 0:
            raise ValueError(f"chunk_margin must be non-negative, got {chunk_margin}.")
        self._binsizes = sorted({int(b) for b in binsizes})
        if any(b < 1 for b in self._binsizes):
            raise ValueError(f"Bin sizes must be positive, got {self._binsizes}.")
        self._loader = loader
        self._shape = tuple(int(s) for s in shape)
        self._reference = reference
        self._name = name
        self._chunk_size = chunk_size
        self._chunk_margin = chunk_margin
        self._chunks = _ChunkCache(chunk_cache_size)
        self._images: dict[int, ip.LazyImgArray] = {}
        self._cache_stack: ExitStack | None = None
        self._lock = threading.Lock()
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
    def tomogram_center(self) -> NDArray[np.float64]:
        """Origin of the tomogram space of the tilt series in nm."""
        return self._loader.tomogram_center

    @property
    def reference(self) -> ip.ImgArray | None:
        """The coarse tomogram for visualization, if given."""
        return self._reference

    @property
    def reference_binsize(self) -> int | None:
        """Bin size of the coarse tomogram for visualization, if given."""
        if self._reference is None:
            return None
        return round(self._reference.scale.x / self.scale)

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
        reconstructed chunks are released. The reference is not updated.
        """
        with self._lock:
            self._loader = self._loader.with_tilt_series(tilt_series)
            self._chunks.clear()
        self._unsaved_tilt_series = True

    def set_reference(self, reference: ip.ImgArray) -> None:
        """Update the coarse tomogram for visualization.

        The new reference must be of the same shape and bin size as the current one.
        """
        if self._reference is not None:
            if reference.shape != self._reference.shape:
                raise ValueError(
                    f"Shape of the reference must be {self._reference.shape}, got "
                    f"{reference.shape}."
                )
            binsize = round(reference.scale.x / self.scale)
            if binsize != self.reference_binsize:
                raise ValueError(
                    f"Bin size of the reference must be {self.reference_binsize}, got "
                    f"{binsize}."
                )
        self._reference = reference
        self._unsaved_reference = True

    def reconstruct_reference(
        self, tilt_series: TiltSeries | None = None
    ) -> ip.ImgArray:
        """Reconstruct the coarse tomogram from the tilt series.

        The tomogram is reconstructed in the same shape and bin size as the current
        reference, in the same way as `prep_reference_tomogram` (light background).
        The current tilt series is used by default.
        """
        if self._reference is None:
            raise ValueError("The tilt series has no reference tomogram.")
        binsize = self.reference_binsize
        assert binsize is not None
        loader = self._loader
        if tilt_series is not None:
            loader = loader.with_tilt_series(tilt_series)
        loader = loader.binning(binsize)
        img = loader.reconstruct_tomogram(self._reference.shape, dtype=np.float32)
        img = ip.asarray(img, axes="zyx", name=self._reference.name)
        return img.set_scale(zyx=self.scale * binsize, unit="nm")

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


def read_tilt_series(path: str | Path, scale: nm | None = None) -> TiltSeries:
    """Read a tilt series.

    Parameters
    ----------
    path : path-like
        A JSON file of `acryo.projection.TiltSeries` (the image paths are relative to
        the file), an IMOD directive file (.edf) or a STAR file of RELION 5 tilt series.
    scale : float, optional
        Pixel size of the tilt images in nm. If not given, the pixel size in the file
        is used. Required for IMOD, which does not store the pixel size.
    """
    from acryo.projection import TiltSeries

    path = Path(path)
    if path.suffix == ".json":
        ts = TiltSeries.from_json(path.read_text())
        if ts.image_path is not None:
            ts = ts.replace(image_path=_resolve_image_path(ts.image_path, path.parent))
        if scale is not None:
            ts = ts.replace(scale=scale)
    elif path.suffix == ".edf":
        if scale is None:
            raise ValueError("Pixel size ('scale') is required for IMOD tilt series.")
        ts = TiltSeries.from_etomo_directory(
            path.parent, scale=scale, local_alignment=True
        )
    elif path.suffix == ".star":
        ts = TiltSeries.from_relion_tilt_series_star(path, scale=scale)
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


def reconstruct_coarse_tomogram(
    tilt_series: TiltSeries,
    binsize: int,
    shape: tuple[int, int, int] | None,
    tomo_hand: int,
) -> ip.ImgArray:
    """Reconstruct a coarse tomogram as `TiltSeriesSource.reconstruct_reference`."""
    from acryo import BackProjectionLoader

    if shape is None:
        raise ValueError("Shape of the coarse tomogram is unknown.")
    if tilt_series.scale is None:
        raise ValueError("Pixel size of the tilt series is not set.")
    loader = BackProjectionLoader(tilt_series, tomo_hand=tomo_hand).binning(binsize)
    img = loader.reconstruct_tomogram(tuple(shape), dtype=np.float32)
    img = ip.asarray(img, axes="zyx")
    return img.set_scale(zyx=tilt_series.scale * binsize, unit="nm")


def prep_reference_tomogram(
    path: str | Path,
    scale: nm | None,
    bin_size: int,
    tomo_hand: int,
    height: int,
    save_coarse_tomogram: bool = True,
) -> ip.ImgArray:
    from acryo import BackProjectionLoader

    path = Path(path)
    loader = BackProjectionLoader(read_tilt_series(path, scale), tomo_hand=tomo_hand)
    loader_binned = loader.binning(bin_size)
    if save_coarse_tomogram:
        save_path = path.with_name(f"{path.stem}_rec_bin{bin_size}.mrc")
        if save_path.exists():
            save_path.unlink()
    else:
        save_path = None
    recon = loader_binned.reconstruct_tomogram((height, -1, -1), save_path=save_path)
    if save_coarse_tomogram:
        return ip.imread(save_path)
    return ip.asarray(recon, axes="zyx").set_scale(zyx=scale * bin_size, unit="nm")


def infer_reference_binsize(
    reference_scale: nm, scale: nm, tolerance: float = 0.02
) -> int:
    """Infer the integer bin size of a reference tomogram from the pixel sizes."""
    binsize = round(reference_scale / scale)
    if binsize < 1 or abs(reference_scale / (binsize * scale) - 1) > tolerance:
        raise ValueError(
            f"Pixel size of the reference tomogram ({reference_scale:.4f} nm) is not an "
            f"integer multiple of the pixel size of the tilt series ({scale:.4f} nm). "
            "Please specify the bin size of the reference tomogram explicitly."
        )
    return binsize


def default_tomogram_center(
    reference_shape: Sequence[int], reference_binsize: int, scale: nm
) -> tuple[float, float, float]:
    """Origin of the tomogram space of a tilt series in the tomogram coordinates."""
    offset = 0
    nz, ny, nx = (n // 2 * reference_binsize + offset for n in reference_shape)
    return (nz * scale, ny * scale, nx * scale)


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
