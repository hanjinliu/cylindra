from __future__ import annotations

from typing import Iterable, Any, overload, TYPE_CHECKING
from pathlib import Path
import warnings
import numpy as np
import impy as ip

from cylindra.const import nm, GlobalVariables as GVar

if TYPE_CHECKING:
    from typing_extensions import Self
    from acryo import Molecules, SubtomogramLoader


class Tomogram:
    """
    Lazy-loading/multi-scale tomogram object. 
    
    It is always connected to a 3D image but processed lazily. Thus you can create
    a lot of Tomogram objects without MemoryError. Subtomograms are temporarily 
    loaded into memory via cache map. Once memory usage exceed certain amount, the
    subtomogram cache will automatically deleted from the old ones.
    """
    _image: ip.LazyImgArray
    _multiscaled: list[tuple[int, ip.ImgArray]]

    def __init__(self):
        self._metadata: dict[str, Any] = {}
        self._image = None
        self._multiscaled = []

    def __hash__(self) -> int:
        """Use unsafe hash."""
        return id(self)
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        shape = str(self._image.shape).lstrip("AxesShape")
        return f"{self.__class__.__name__}{shape}"
    
    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata relevant to the tomogram."""
        return self._metadata
    
    @metadata.setter
    def metadata(self, v: dict[str, Any]):
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
    def from_image(
        cls,
        img: ip.ImgArray | ip.LazyImgArray,
        *,
        scale: float | None = None,
        binsize: int | Iterable[int] = (),
    ):
        """
        Construct a Tomogram object from a image array.
        
        Parameters
        ----------
        img : array-like
            Input image.
        scale : float, optional
            Pixel size in nm. If not given, will try to read from image header.
        binsize : int or iterable of int, optional
            Binsize to generate multiscale images. If not given, will not generate.
        
        Returns
        -------
        Tomogram
            Tomogram object with the image that has just been read and multi-scales.
        """
        self = cls()
        if type(img) is np.ndarray:
            img = ip.asarray(img, axes="zyx")

        if scale is not None:
            img.set_scale(xyz=scale)
        else:
            if (abs(img.scale.z - img.scale.x) > 1e-3
                or abs(img.scale.z - img.scale.y) > 1e-3
            ):
                raise ValueError(f"Uneven scale: {img.scale}.")
        
        self._set_image(img)
        if source := img.source:
            self._metadata["source"] = source.resolve()
        self._metadata["scale"] = scale
        
        if isinstance(binsize, int):
            binsize = [binsize]
        for b in sorted(binsize):
            self.add_multiscale(b)
        return self
    
    @classmethod
    def imread(
        cls, 
        path: str | Path,
        *, 
        scale: float = None,
        binsize: int | Iterable[int] = (),
    ) -> Self:
        """
        Read a image as a dask array.
        
        Parameters
        ----------
        path : path-like
            Path to the image file.
        scale : float, optional
            Pixel size in nm. If not given, will try to read from image header.
        binsize : int or iterable of int, optional
            Binsize to generate multiscale images. If not given, will not generate.
        
        Returns
        -------
        Tomogram
            Tomogram object with the image that has just been read and multi-scales.
        """
        img = ip.lazy_imread(path, chunks=GVar.daskChunk, name="tomogram").as_float()
        return cls.from_image(img, scale=scale, binsize=binsize)
        
    @property
    def image(self) -> ip.LazyImgArray:
        """Tomogram image data."""
        return self._image
    
    @property
    def multiscaled(self) -> list[tuple[int, ip.ImgArray]]:
        """Get all multi-scale factor and the corresponding multiscaled images."""
        return self._multiscaled
    
    def _set_image(self, img: ip.LazyImgArray | np.ndarray) -> None:
        if isinstance(img, ip.LazyImgArray):
            _img = img.as_float()
        elif isinstance(img, np.ndarray):
            if img.ndim != 3:
                raise ValueError("Can only set 3-D image.")
            _img = ip.aslazy(img, dtype=np.float32, axes="zyx", chunks=GVar.daskChunk)
            if isinstance(img, ip.ImgArray):
                _img.set_scale(img)
        else:
            raise TypeError(f"Cannot set type {type(img)} as an image.")
        if (abs(_img.scale.z - _img.scale.x) > 1e-4
            or abs(_img.scale.z - _img.scale.y) > 1e-4):
            raise ValueError("Uneven scale.")
        self.scale = _img.scale.x
        self._image = _img
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
        # iterate from the larger bin size
        for _b, _img in reversed(self._multiscaled):
            if binsize == _b:
                warnings.warn(
                    f"Binsize {binsize} already exists in multiscale images. "
                    "Skip binning process.",
                    UserWarning,
                )
                return _img
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
    
    def _get_multiscale_or_original(self, binsize: int) -> ip.ImgArray | ip.LazyImgArray:
        """Get multiscaled image of given binsize but use original one if needed."""
        if binsize > 1:
            out = self.get_multiscale(binsize)
        else:
            try:
                out = self.get_multiscale(1)
            except ValueError:
                out = self.image
        return out
    
    def multiscale_translation(self, binsize: int) -> nm:
        """Get lateral translation of binned image in nm."""
        return (binsize - 1) / 2 * self.scale
    
    def invert(self) -> Self:
        """Invert tomogram intensities **in-place**."""
        img_inv = -self.image
        img_inv.release()
        self._set_image(img_inv)
        for i in range(len(self._multiscaled)):
            _b, _img = self._multiscaled[i]
            self._multiscaled[i] = (_b, -_img)
        return self
    
    def lowpass_filter(self, cutoff: float) -> Self:
        """Low-pass filtering the original image **in-place**"""
        if 0 < cutoff < 0.866:
            self.image.tiled_lowpass_filter(cutoff, update=True, overlap=32)
            self.image.release()
        return self
    
    def get_subtomogram_loader(
        self,
        mole: Molecules,
        shape: tuple[nm, nm, nm] | None = None,
        binsize: int = 1,
        order: int = 1,
    ) -> SubtomogramLoader:
        """Create a subtomogram loader from molecules."""
        from acryo import SubtomogramLoader
        
        if binsize == 1:
            try:
                img = self.get_multiscale(1)
            except ValueError:
                img = self.image
        else:
            tr = -self.multiscale_translation(binsize)
            mole = mole.translate([tr, tr, tr])
            img = self.get_multiscale(binsize)
        
        kwargs = dict(
            order=order,
            scale=self.scale*binsize,
        )
        if shape is not None:
            kwargs["output_shape"] = tuple(self.nm2pixel(shape, binsize=binsize))
        return SubtomogramLoader(
            img.value,
            mole,
            **kwargs,
        )