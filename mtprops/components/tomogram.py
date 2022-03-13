from __future__ import annotations

from typing import Iterable, Any, overload, TYPE_CHECKING
from pathlib import Path
import warnings
import numpy as np
import impy as ip

from ..const import nm, GVar

if TYPE_CHECKING:
    from typing_extensions import Self
    from .molecules import Molecules
    from .loader import SubtomogramLoader


class Tomogram:
    """
    Tomogram object. It always connected to a 3D image but processed lazily. Thus
    you can create a lot of MtTomogram objects without MemoryError. Subtomograms
    are temporarily loaded into memory via cache map. Once memory usage exceed
    certain amount, the subtomogram cache will automatically deleted from the old
    ones.
    """
    _image: ip.LazyImgArray
    _multiscaled: list[tuple[int, ip.ImgArray]]

    def __init__(self):
        self._metadata: dict[str, Any] = {}
        self._image = None
        self._multiscaled = []

    def __hash__(self) -> int:
        return id(self)
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        shape = str(self._image.shape).lstrip("AxesShape")
        return f"{self.__class__.__name__}{shape!r}"
    
    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata relevant to the tomogram."""
        return self._metadata
    
    @metadata.setter
    def metadata(self, v: dict):
        if not isinstance(v, dict):
            raise TypeError(f"Cannot set type {type(v)} as a metadata.")
        self._metadata = v
    
    @property
    def source(self) -> Path:
        """Path to the source file."""
        source = self.metadata.get("source", None)
        if source is None:
            raise ValueError("Source file is unknown.")
        return Path(source)
    
    @classmethod
    def imread(
        cls, 
        path: str,
        *, 
        scale: float = None,
    ) -> Self:
        """Read a image as a dask array."""
        self = cls()
        img = ip.lazy_imread(path, chunks=GVar.daskChunk).as_float()
        if scale is not None:
            img.set_scale(xyz=scale)
        else:
            if (abs(img.scale.z - img.scale.x) > 1e-3
                or abs(img.scale.z - img.scale.y) > 1e-3
            ):
                raise ValueError(f"Uneven scale: {img.scale}.")
        
        self._set_image(img)
        self._metadata["source"] = Path(path).resolve()
        self._metadata["scale"] = scale
        return self
        
    @property
    def image(self) -> ip.LazyImgArray:
        """Tomogram image data."""
        return self._image
    
    @property
    def multiscaled(self) -> list[tuple[int, ip.ImgArray]]:
        return self._multiscaled
    
    def _set_image(self, img: ip.LazyImgArray | np.ndarray) -> None:
        if isinstance(img, ip.LazyImgArray):
            img = img.as_float()
        elif isinstance(img, np.ndarray):
            if img.ndim != 3:
                raise ValueError("Can only set 3-D image.")
            img = ip.aslazy(img, dtype=np.float32, axes="zyx", chunks=GVar.daskChunk)
        else:
            raise TypeError(f"Cannot set type {type(img)} as an image.")
        if (abs(img.scale.z - img.scale.x) > 1e-4
            or abs(img.scale.z - img.scale.y) > 1e-4):
            raise ValueError("Uneven scale.")
        self.scale = img.scale.x
        self._image = img
        return None
    
    @overload
    def nm2pixel(self, value: nm, binsize: int = 1) -> int:
        ...
        
    @overload
    def nm2pixel(self, value: Iterable[nm], binsize: int = 1) -> np.ndarray:
        ...
    
    def nm2pixel(self, value, binsize: int = 1):
        """
        Convert nm float value into pixel value. Useful for conversion from 
        coordinate to pixel position.

        Returns
        -------
        np.ndarray or int
            Pixel position.
        """        
        pix = np.round(np.asarray(value)/self.scale/binsize).astype(np.int16)
        if np.isscalar(value):
            pix = int(pix)
        return pix

    def add_multiscale(self, binsize: int) -> ip.ImgArray:
        """Add new multiscaled image of given binsize."""
        with ip.silent():
            # iterate from the larger bin size
            for _b, _img in reversed(self._multiscaled):
                if binsize == _b:
                    warnings.warn(
                        f"Binsize {binsize} already exists in multiscale images. "
                        "Skip binning process.",UserWarning
                    )
                    return
                if binsize % _b == 0:
                    imgb = _img.binning(binsize//_b, check_edges=False)
                    break
            else:
                imgb = self.image.binning(binsize, check_edges=False).compute()
        self._multiscaled.append((binsize, imgb))
        self._multiscaled.sort(key=lambda x: x[0])
        return imgb
    
    def get_multiscale(self, binsize: int, add: bool = False) -> ip.ImgArray:
        """Get multiscaled image of given binsize."""
        for _b, _img in self._multiscaled:
            if _b == binsize:
                return _img
        if add:
            return self.add_multiscale(binsize)
        else:
            raise ValueError(f"Multiscale = {binsize} not found.")
    
    def multiscale_translation(self, binsize: int) -> nm:
        """Get lateral translation of binned image in nm."""
        return (binsize - 1) / 2 * self.scale
    
    def invert(self) -> Self:
        """Invert tomogram intensities **in-place**."""
        img_inv = -self.image
        with ip.silent():
            img_inv.release()
        self._set_image(img_inv)
        for i in range(len(self._multiscaled)):
            _b, _img = self._multiscaled[i]
            self._multiscaled[i] = (_b, -_img)
        return self
    
    def lowpass_filter(self, cutoff: float) -> Self:
        """Low-pass filtering the original image **in-place**"""
        if 0 < cutoff < 0.866:
            with ip.silent():
                self.image.tiled_lowpass_filter(cutoff, update=True, overlap=32)
                self.image.release()
        return self
    
    def get_subtomogram_loader(
        self,
        mole: Molecules,
        shape: tuple[nm, nm, nm], 
        binsize: int = 1,
        order: int = 1,
        chunksize: int = 128,
    ) -> SubtomogramLoader:
        """Create a subtomogram loader from molecules."""
        from .loader import SubtomogramLoader
        output_shape = tuple(self.nm2pixel(shape, binsize=binsize))
        if binsize == 1:
            try:
                img = self.get_multiscale(1)
            except ValueError:
                img = self.image
        else:
            tr = -self.multiscale_translation(binsize)
            mole = mole.translate([tr, tr, tr])
            img = self.get_multiscale(binsize)
        return SubtomogramLoader(
            img, mole, output_shape=output_shape, order=order, chunksize=chunksize
        )