from __future__ import annotations
from typing import Iterable, Iterator, NamedTuple
import random
import warnings
import weakref
import numpy as np
import impy as ip
from dask import array as da
from .spline import build_local_cartesian
from .utils import map_coordinates, multi_map_coordinates
from .molecules import Molecules
from .utils import no_verbose


class SubtomogramSampler:
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
    
    @property
    def image_ref(self) -> ip.ImgArray | ip.LazyImgArray:
        image = self._image_ref()
        if image is None:
            raise ValueError("No tomogram found.")
        return image
    
    @image_ref.setter
    def image_ref(self, image: ip.ImgArray | ip.LazyImgArray):
        # Sampler is fully depend on an image. We'd better use a weakref.
        self._image_ref = weakref.ref(image)
    
    def __iter__(self) -> Iterator[ip.ImgArray]:  # axes: zyx
        for subvols in self.iter_chunks():
            for subvol in subvols:
                yield subvol
    
    def iter_chunks(self) -> Iterator[ip.ImgArray]:  # axes: pzyx
        centers = self.molecules.pos
        ds_list = self.molecules.y
        image = self.image_ref
        scale = image.scale
        
        with no_verbose():
            for coords in _iter_coordinate(centers/scale, ds_list, 
                                           self.output_shape, chunksize=self.chunksize):
                subvols = np.stack(multi_map_coordinates(image, coords, cval=np.mean), axis=0)
                subvols = ip.asarray(subvols, axes="pzyx")
                subvols.set_scale(xyz=scale)
                yield subvols
    
    def to_stack(self) -> ip.ImgArray:
        images = list(self)
        return np.stack(images, axis="p")
    
    def align(
        self,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None
    ):
        if template is None:
            raise NotImplementedError("Template image is needed.")
        if mask is not None:
            mask = 1
        if template.shape != self.output_shape:
            warnings.warn(
                f"'output_shape' of {self.__class__.__name__} object {self.output_shape!r} "
                f"differs from the shape of template image {template.shape!r}. 'output_shape' "
                "is updated.",
                UserWarning
            )
            self.output_shape = template.shape
        
        image = self.image_ref
        scale = image.scale
        shifts: list[np.ndarray] = []
        pre_alignment = np.zeros_like(template)
        
        with no_verbose():
            template_ft = (template * mask).fft()
            for subvol in self:
                shift = ip.ft_pcc_maximum((subvol*mask).fft(), template_ft)
                shifts.append(shift)
                pre_alignment += subvol
        
        mole_aligned = self.molecules.translate(np.stack(shifts, axis=0)/scale)
        pre_alignment = ip.asarray(pre_alignment/len(shifts), axes="zyx", name="Avg")
        pre_alignment.set_scale(xyz=scale)
        
        return mole_aligned, pre_alignment

    def fsc(self, mask=None, seed=0):
        random.seed(seed)
        image = self.image_ref
        
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
                sum_images[0] += sum(subsets[:lc], axis=0)
                sum_images[1] += sum(subsets[lc:], axis=0)
        
        if next_set == 1:
            random.shuffle(subsets)
            lc = len(subsets) // 2
            sum_images[0] += sum(subsets[:lc], axis=0)
            sum_images[1] += sum(subsets[lc:], axis=0)
            
        random.seed(None)
        fsc = ip.fsc(ip.asarray(sum_images[0], axes="zyx"),
                     ip.asarray(sum_images[1], axes="zyx"),
                     mask=mask)
        return fsc
        
        

def _iter_coordinate(
    centers: Iterable[np.ndarray],
    ds_list: Iterable[np.ndarray],
    size: tuple[int, ...],
    chunksize: int
) -> Iterator[np.ndarray]:
    all_coords = []
    count = 0
    for center, ds in zip(centers, ds_list):
        coords = build_local_cartesian(size, ds, center)
        all_coords.append(coords)
        if count == chunksize:
            out = np.stack(all_coords, axis=0)
            out = np.moveaxis(out, -1, 1)
            yield out
            all_coords = []
            count = 0
        else:
            count += 1
    if count != 0:
        out = np.stack(all_coords, axis=0)
        out = np.moveaxis(out, -1, 1)
        yield out