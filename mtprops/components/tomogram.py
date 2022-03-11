from __future__ import annotations

from typing import Iterable, Any, overload, TYPE_CHECKING
from pathlib import Path
import numpy as np
import impy as ip

from ..const import nm, GVar

if TYPE_CHECKING:
    from typing_extensions import Self


class Tomogram:
    """
    Tomogram object. It always connected to a 3D image but processed lazily. Thus
    you can create a lot of MtTomogram objects without MemoryError. Subtomograms
    are temporarily loaded into memory via cache map. Once memory usage exceed
    certain amount, the subtomogram cache will automatically deleted from the old
    ones.
    """
    _image: ip.LazyImgArray

    def __init__(self):
        self._metadata: dict[str, Any] = {}

    def __hash__(self) -> int:
        return id(self)
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        shape = str(self._image.shape).lstrip("AxesShape")
        return f"{self.__class__.__name__}{shape!r}"
    
    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
    
    @metadata.setter
    def metadata(self, v: dict):
        if not isinstance(v, dict):
            raise TypeError(f"Cannot set type {type(v)} as a metadata.")
        self._metadata = v
    
    @property
    def source(self) -> Path:
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
        
        self = cls()
        img = ip.lazy_imread(path, chunks=GVar.daskChunk).as_float()
        if scale is not None:
            img.set_scale(xyz=scale)
        else:
            if (abs(img.scale.z - img.scale.x) > 1e-4
                or abs(img.scale.z - img.scale.y) > 1e-4):
                raise ValueError("Uneven scale.")
        
        self._set_image(img)
        self._metadata["source"] = Path(path).resolve()
        return self
        
    @property
    def image(self) -> ip.LazyImgArray:
        """Tomogram image data."""
        return self._image
    
    def _set_image(self, img: ip.LazyImgArray | np.ndarray) -> None:
        if isinstance(img, ip.LazyImgArray):
            pass
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
    def nm2pixel(self, value: nm) -> int:
        ...
        
    @overload
    def nm2pixel(self, value: Iterable[nm]) -> np.ndarray:
        ...
    
    def nm2pixel(self, value):
        """
        Convert nm float value into pixel value. Useful for conversion from 
        coordinate to pixel position.

        Returns
        -------
        np.ndarray or int
            Pixel position.
        """        
        pix = np.round(np.asarray(value)/self.scale).astype(np.int16)
        if np.isscalar(value):
            pix = int(pix)
        return pix
