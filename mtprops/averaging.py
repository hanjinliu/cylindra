from __future__ import annotations
from typing import Any, Callable, Iterator, Iterable, Union
import random
import warnings
import weakref
import numpy as np
import impy as ip
from dask import array as da
from .utils import multi_map_coordinates, no_verbose
from .molecules import Molecules
from .const import nm

RangeLike = Union[
    list[float],
    np.ndarray,
    tuple[float, float, int],
]

class SubtomogramLoader:
    """
    A class for efficient loading of subtomograms.
    
    Parameters
    ----------
    image : ImgArray, LazyImgArray, np.ndarray or da.core.Array
        Tomogram image.
    mole : Molecules
        Molecules object that defines position and rotation of subtomograms.
    output_shape : int or tuple of int
        Shape of output subtomograms.
    chunksize : int, optional
        Chunk size used when loading subtomograms.
    """
    
    def __init__(
        self, 
        image: ip.ImgArray | ip.LazyImgArray | np.ndarray | da.core.Array, 
        mole: Molecules,
        output_shape: int | tuple[int, int, int],
        chunksize: int | None = None,
    ) -> None:
        ndim = 3
        if not isinstance(image, (ip.ImgArray, ip.LazyImgArray)):
            if isinstance(image, np.ndarray) and image.ndim == ndim:
                image = ip.asarray(image, axes="zyx")
            elif isinstance(image, da.core.Array) and image.ndim == ndim:
                image = ip.LazyImgArray(image, axes="zyx")
            else:
                raise TypeError("'image' must be a 3D numpy ndarray like object.")
        self.image_ref = image
        self.molecules = mole
        self.chunksize = chunksize or 1
        if isinstance(output_shape, int):
            output_shape = (output_shape,) * ndim
        self.output_shape = output_shape
        self.image_avg: ip.ImgArray | None = None
    
    def __repr__(self) -> str:
        shape = self.image_ref.shape
        mole_repr = repr(self.molecules)
        return f"{self.__class__.__name__}(tomogram shape: {shape}, molecules: {mole_repr})"
    
    @property
    def image_ref(self) -> ip.ImgArray | ip.LazyImgArray:
        """Return tomogram."""
        image = self._image_ref()
        if image is None:
            raise ValueError("No tomogram found.")
        return image
    
    @image_ref.setter
    def image_ref(self, image: ip.ImgArray | ip.LazyImgArray):
        """Set tomogram as a weak reference."""
        self._image_ref = weakref.ref(image)
    
    @property
    def scale(self) -> nm:
        return self.image_ref.scale.x
    
    def __iter__(self) -> Iterator[ip.ImgArray]:  # axes: zyx
        """Generate each subtomogram."""
        for subvols in self.iter_chunks():
            for subvol in subvols:
                yield subvol
    
    def iter_chunks(self) -> Iterator[ip.ImgArray]:  # axes: pzyx
        """Generate subtomogram list chunk-wise."""
        image = self.image_ref
        scale = image.scale.x
        
        with no_verbose():
            for coords in self.molecules.iter_cartesian(self.output_shape, scale, self.chunksize):
                subvols = np.stack(multi_map_coordinates(image, coords, cval=np.mean), axis=0)
                subvols = ip.asarray(subvols, axes="pzyx")
                subvols.set_scale(image)
                yield subvols
    
    def to_stack(self) -> ip.ImgArray:
        """Create a 4D image stack of all the subtomograms."""
        images = list(self)
        return np.stack(images, axis="p")

    def subset(self, spec: slice | list[int] | np.ndarray) -> SubtomogramLoader:
        """
        Generate a new SubtomogramLoader object with a subset of subtomograms.

        This method works similar to the ``subset`` method in ``Molecules``.
        
        Parameters
        ----------
        spec : slice
            Specifier that defines which subtomograms will be used. Any objects that numpy
            slicing are defined are supported. For instance, ``[2, 3, 5]`` means the 2nd,
            3rd and 5th subtomograms will be used (zero-indexed), and ``slice(10, 20)``
            means the 10th to 19th subtomograms will be used.

        Returns
        -------
        SubtomogramLoader
            Instance with a subset of subtomograms.
        """
        mole = self.molecules.subset(spec)
        return self.__class__(self.image_ref, mole, self.output_shape, self.chunksize)

    def average(
        self,
        *,
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> ip.ImgArray:
        """
        Average all the subtomograms.
        
        The averaged image will be stored in ``self.averaged_image``. The size of subtomograms
        is determined by the ``self.output_shape`` attribute.

        Parameters
        ----------
        callback : callable, optional
            If given, ``callback(self)`` will be called for each iteration of subtomogram loading.

        Returns
        -------
        ImgArray
            Averaged image.
        """
        aligned = np.zeros(self.output_shape, dtype=np.float32)
        n = 0
        if callback is None:
            callback = lambda x: None
        with no_verbose():
            for subvol in self:
                aligned += subvol.value
                n += 1
                callback(self)
        self.image_avg = ip.asarray(aligned / n, name="Avg", axes="zyx")
        self.image_avg.set_scale(self.image_ref)
        return self.image_avg
        
    def align(
        self,
        *,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        max_shifts: int | tuple[int, int, int] = 4,
        rotations: RangeLike | None = None,
        cutoff: float = 0.5,
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> SubtomogramLoader:
        
        # normalize input
        if template is None:
            raise NotImplementedError("Template image is needed.")
        if mask is not None:
            mask = 1
        if template.shape != self.output_shape:
            warnings.warn(
                f"'output_shape' of {self.__class__.__name__} object {self.output_shape!r} "
                f"differs from the shape of template image {template.shape!r}. 'output_shape' "
                "is updated.",
                UserWarning,
            )
            self.output_shape = template.shape
        if rotations is not None:
            if isinstance(rotations, tuple):
                start, stop, step = rotations
                num, res = divmod((stop - start), step)
                rotations = np.linspace(start + res/2, stop-res/2, num)
            else:
                rotations = np.asarray(rotations)
                
        if callback is None:
            callback = lambda x: None
        
        local_shifts: list[np.ndarray] = []  # shift in local Cartesian
        pre_alignment = np.zeros_like(template.value)
        
        with no_verbose():
            template_ft = (template.lowpass_filter(cutoff=cutoff) * mask).fft(shift=False)
            for subvol in self:
                input_subvol = subvol.lowpass_filter(cutoff=cutoff) * mask
                shift = ip.ft_pcc_maximum(
                    input_subvol.fft(shift=False),
                    template_ft, 
                    upsample_factor=20, 
                    max_shifts=max_shifts
                )
                local_shifts.append(shift)
                if self.image_avg is None:
                    pre_alignment += subvol
                callback(self)
                
        shifts = self.molecules._rotator.apply(np.stack(local_shifts, axis=0) * self.scale)
        mole_aligned = self.molecules.translate(shifts)
        out = self.__class__(self.image_ref, mole_aligned, self.output_shape, self.chunksize)
        
        if self.image_avg is None:
            pre_alignment = ip.asarray(pre_alignment/len(shifts), axes="zyx", name="Avg")
            pre_alignment.set_scale(self.image_ref)
            self.image_avg = pre_alignment
        
        return out

    def fsc(
        self,
        mask: ip.ImgArray | None = None,
        seed: int | float | str | bytes | bytearray | None = 0,
        ) -> np.ndarray:
        random.seed(seed)
        
        subsets: list[np.ndarray] = []
        sum_images = (np.zeros(self.output_shape, dtype=np.float32),
                      np.zeros(self.output_shape, dtype=np.float32))
        next_set = 0
        for subvols in self.iter_chunks():
            subsets.extend(list(subvols.value))
            next_set = 1 - next_set
            if next_set == 0:
                random.shuffle(subsets)
                lc = len(subsets) // 2
                sum_images[0] += sum(subsets[:lc])
                sum_images[1] += sum(subsets[lc:])
        
        if next_set == 1:
            random.shuffle(subsets)
            lc = len(subsets) // 2
            sum_images[0] += sum(subsets[:lc])
            sum_images[1] += sum(subsets[lc:])
            
        random.seed(None)
        fsc = ip.fsc(ip.asarray(sum_images[0], axes="zyx"),
                     ip.asarray(sum_images[1], axes="zyx"),
                     mask=mask)
        
        if self.image_avg is None:
            self.image_avg = ip.asarray(sum_images[0] + sum_images[1], axes="zyx", name="Avg")
            self.image_avg.set_scale(self.image_ref)
            
        return np.asarray(fsc)
