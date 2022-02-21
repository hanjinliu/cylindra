from __future__ import annotations
import itertools
from typing import Any, Callable, Generator, Iterator, Iterable, Union
import random
import warnings
import weakref
from scipy.spatial.transform import Rotation
import numpy as np
import impy as ip
from dask import array as da
from .utils import multi_map_coordinates, no_verbose
from .molecules import Molecules, from_euler
from .const import nm

RangeLike = tuple[float, int]
Ranges = Union[RangeLike, tuple[RangeLike, RangeLike, RangeLike]]

def _normalize_a_range(rng: RangeLike) -> tuple[float, int]:
    if len(rng) != 2:
        raise TypeError("Range must be defined by (float, int).")
    max_rot, step = rng
    return float(max_rot), float(step)
        
def _normalize_ranges(rng: Ranges) -> tuple[tuple[float, int], tuple[float, int], tuple[float, int]]:
    if isinstance(rng, tuple) and isinstance(rng[0], tuple):
        return tuple(_normalize_a_range(r) for r in rng)
    else:
        rng = _normalize_a_range(rng)
        return (rng,) * 3


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
    
    def __len__(self) -> int:
        """Return the number of subtomograms."""
        return self.molecules.pos.shape[0]
    
    def __iter__(self) -> Iterator[ip.ImgArray]:  # axes: zyx
        """Generate each subtomogram."""
        return self.iter_subtomograms()
    
    def iter_subtomograms(
        self,
        rotators: Iterable[Rotation] | None = None,
        order: int = 3
    ) -> Iterator[ip.ImgArray]:  # axes: zyx or azyx
        if rotators is None:
            iterator = self._iter_chunks(order=order)
        else:
            iterator = self._iter_chunks_with_rotation(rotators=rotators, order=order)
            
        for subvols in iterator:
            for subvol in subvols:
                yield subvol
    
    def _iter_chunks(self, order: int = 3) -> Iterator[ip.ImgArray]:  # axes: pzyx
        """Generate subtomogram list chunk-wise."""
        image = self.image_ref
        scale = image.scale.x
        
        with no_verbose():
            for coords in self.molecules.iter_cartesian(self.output_shape, scale, self.chunksize):
                subvols = np.stack(
                    multi_map_coordinates(image, coords, order=order, cval=np.mean),
                    axis=0,
                    )
                subvols = ip.asarray(subvols, axes="pzyx")
                subvols.set_scale(image)
                yield subvols
    
    def _iter_chunks_with_rotation(
        self, 
        rotators: Iterable[Rotation], 
        order: int = 3
    ) -> Iterator[ip.ImgArray]:  # axes: pazyx
        image = self.image_ref
        scale = image.scale.x
        mole_list = [self.molecules.rotate_by(rot) for rot in rotators]
        nrot = len(rotators)
        chunksize = max(self.chunksize//nrot, 1)
        iterators = [mole.iter_cartesian(self.output_shape, scale, chunksize) for mole in mole_list]
        
        with no_verbose():
            for coords_list in zip(*iterators):
                coords = np.concatenate(coords_list, axis=0)
                subvols = np.stack(
                    multi_map_coordinates(image, coords, order=order, cval=np.mean),
                    axis=0,
                    )
                subvols = subvols.reshape((-1, nrot) + self.output_shape)
                subvols = ip.asarray(subvols, axes="pazyx")
                subvols.set_scale(image)
                yield subvols
    
    
    def to_stack(self) -> ip.ImgArray:
        """Create a 4D image stack of all the subtomograms."""
        images = list(self)
        return np.stack(images, axis="p")


    def subset(self, spec: int | slice | list[int] | np.ndarray) -> SubtomogramLoader:
        """
        Generate a new SubtomogramLoader object with a subset of subtomograms.

        This method works similar to the ``subset`` method in ``Molecules``.
        
        Parameters
        ----------
        spec : int, slice, list of int or np.ndarray
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

    def iter_average(self, order: int = 3) -> Generator[ip.ImgArray, None, ip.ImgArray]:
        aligned = np.zeros(self.output_shape, dtype=np.float32)
        n = 0
        with no_verbose():
            for subvol in self.iter_subtomograms(order=order):
                aligned += subvol.value
                n += 1
                yield aligned
        self.image_avg = ip.asarray(aligned / n, name="Avg", axes="zyx")
        self.image_avg.set_scale(self.image_ref)
        return self.image_avg
    
    def average(
        self,
        *,
        order: int = 1,
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> ip.ImgArray:
        """
        Average all the subtomograms.
        
        The averaged image will be stored in ``self.averaged_image``. The size of subtomograms
        is determined by the ``self.output_shape`` attribute.

        Parameters
        ----------
        order : int, default is 1
            Order of interpolation. See ``scipy.ndimage.map_coordinates``.
        callback : callable, optional
            If given, ``callback(self)`` will be called for each iteration of subtomogram loading.

        Returns
        -------
        ImgArray
            Averaged image.
        """
        if callback is None:
            callback = lambda x: None
        for i in self.iter_average(order=order):
            callback(self)
        return self.image_avg
    
    def iter_align(
        self,
        *,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        max_shifts: int | tuple[int, int, int] = 4,
        rotations: RangeLike | None = None,
        cutoff: float = 0.5,
        order: int = 1,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, SubtomogramLoader]:
        
        if template is None:
            raise NotImplementedError("Template image is needed.")
        if mask is None:
            mask = 1
        self._check_shape(template)
        
        # Convert rotations into quaternion if given.
        if rotations is not None:
            rotations = _normalize_ranges(rotations)
            angles = []
            for max_rot, step in rotations:
                if step == 0:
                    angles.append(np.zeros(1))
                else:
                    n = int(max_rot / step)
                    angles.append(np.linspace(-n*step, n*step, 2*n + 1))
            
            quat: list[np.ndarray] = []
            for angs in itertools.product(*angles):
                quat.append(from_euler(np.array(angs), "zyx", degrees=True).as_quat())
            if len(quat) == 1:
                rotations = None
            else:
                rotations = np.stack(quat, axis=0)
        
        pre_alignment = np.zeros_like(template.value)
        
        # shift in local Cartesian
        local_shifts = np.zeros((len(self), 3))
        
        # rotation (quaternion) in local Cartesian
        local_rot = np.zeros((len(self), 4))
        local_rot[:, 3] = 1  # identity map in quaternion
        
        with no_verbose():
            template_ft = (template.lowpass_filter(cutoff=cutoff) * mask).fft()
            if rotations is None:
                for i, subvol in enumerate(self.iter_subtomograms(order=order)):
                    input_subvol = subvol.lowpass_filter(cutoff=cutoff) * mask
                    shift = ip.ft_pcc_maximum(
                        input_subvol.fft(),
                        template_ft, 
                        upsample_factor=20, 
                        max_shifts=max_shifts
                    )
                    if self.image_avg is None:
                        pre_alignment += subvol
                    local_shifts[i, :] = shift
                    
                    yield local_shifts[i, :], local_rot[i, :]
                    
            else:
                rotators = [Rotation.from_quat(r) for r in rotations]
                iterator = self.iter_subtomograms(rotators=rotators, order=order)
                for i, subvol_set in enumerate(iterator):
                    corrs: list[float] = []
                    all_shifts: list[np.ndarray] = []
                    for subvol in subvol_set:
                        subvol: ip.ImgArray
                        input_subvol = subvol.lowpass_filter(cutoff=cutoff) * mask
                        shift = ip.ft_pcc_maximum(
                            input_subvol.fft(),
                            template_ft, 
                            upsample_factor=20, 
                            max_shifts=max_shifts
                        )
                        all_shifts.append(shift)
                        shifted_subvol = input_subvol.affine(translation=shift)
                        corr = ip.zncc(shifted_subvol*mask, template*mask)
                        corrs.append(corr)
                    
                    if self.image_avg is None:
                        pre_alignment += subvol_set[0]
                    
                    iopt = np.argmax(corrs)
                    local_shifts[i, :] = all_shifts[iopt]
                    local_rot[i, :] = rotations[iopt]
                    
                    yield local_shifts[i, :], local_rot[i, :]
        
        if self.image_avg is None:
            pre_alignment = ip.asarray(pre_alignment/len(self), axes="zyx", name="Avg")
            pre_alignment.set_scale(self.image_ref)
            self.image_avg = pre_alignment
        
        shifts = self.molecules.rotator.apply(np.stack(local_shifts, axis=0) * self.scale)
        mole_aligned = self.molecules.translate(shifts)
        if rotations is not None:
            mole_aligned = mole_aligned.rotate_by_quaternion(local_rot)
            
        out = self.__class__(self.image_ref, mole_aligned, self.output_shape, self.chunksize)
        return out
        
    def align(
        self,
        *,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        max_shifts: int | tuple[int, int, int] = 4,
        rotations: RangeLike | None = None,
        cutoff: float = 0.5,
        order: int = 1,
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> SubtomogramLoader:        
        """
        Align subtomograms to a template to get high-resolution image.
        
        This method conduct so called "subtomogram averaging". Only shifts and rotations
        are calculated in this method. To get averaged image, you'll have run "average"
        method using the resulting SubtomogramLoader instance.
        
        Parameters
        ----------
        template : ip.ImgArray, optional
            Template image.
        mask : ip.ImgArray, optional
            Mask image. Must in the same shae as the template.
        max_shifts : int or tuple of int, default is (4, 4, 4)
            Maximum shift between subtomograms and template.
        rotations : RangeLike | None, optional
            Rotation between subtomograms and template in external Euler angles.
        cutoff : float, default is 0.5
            Cutoff frequency of low-pass filter applied in each subtomogram.
        order : int, default is 1
            Interpolation order.
        callback : Callable[[SubtomogramLoader], Any], optional
            Callback function that will get called after each iteration.

        Returns
        -------
        SubtomogramLoader
            Refined molecule object is bound.
        """        
        
        if callback is None:
            callback = lambda x: None
        
        align_iter = self.iter_align(
            template=template, 
            mask=mask,
            max_shifts=max_shifts,
            rotations=rotations,
            cutoff=cutoff,
            order=order,
        )
        
        while True:
            try:
                next(align_iter)
                callback(self)
            except StopIteration as mole_aligned:
                break
        
        out = self.__class__(self.image_ref, mole_aligned, self.output_shape, self.chunksize)
        
        return out

    def iter_each_seam(
        self,
        npf: int,
        template: ip.ImgArray,
        mask: ip.ImgArray | None = None,
        load_all: bool = False,
        order: int = 1,
    ) -> Generator[tuple[float, ip.ImgArray, Molecules],
                   None,
                   tuple[np.ndarray, ip.ImgArray, list[Molecules]]]:
        averaged_images: list[ip.ImgArray] = []
        corrs: list[float] = []
        candidates: list[Molecules] = []
        
        if mask is None:
            mask = 1
        
        self._check_shape(template)
        masked_template = template * mask
        
        with no_verbose():
            if load_all:
                subtomograms = np.stack(list(self.iter_subtomograms(order=order)), axis="p")
                
            for pf in range(2*npf):
                _id = np.arange(len(self.molecules))
                res = (_id - pf) // npf
                sl = res % 2 == 0
                candidate = self.subset(sl)
                candidates.append(candidate)
                if not load_all:
                    image_ave = candidate.average(order=order)
                else:
                    image_ave = np.mean(subtomograms[sl], axis=0)
                averaged_images.append(image_ave)
                corr = ip.zncc(image_ave*mask, masked_template)
                corrs.append(corr)
                yield corr, image_ave, candidate
                
        return np.array(corrs), np.stack(averaged_images, axis="p"), candidates
    
    def try_all_seams(
        self,
        npf: int,
        template: ip.ImgArray,
        mask: ip.ImgArray | None = None,
        load_all: bool = False,
        order: int = 1,
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> tuple[np.ndarray, ip.ImgArray, list[Molecules]]:
        if callback is None:
            callback = lambda x: None
        
        seam_iter = self.iter_each_seam(
            npf=npf,
            template=template, 
            mask=mask,
            load_all=load_all,
            order=order,
        )
        
        while True:
            try:
                next(seam_iter)
                callback(self)
            except StopIteration as results:
                break
        
        return results
    
    
    def average_split(
        self, 
        *,
        seed: int | float | str | bytes | bytearray | None = 0,
        order: int = 1,
    ) -> tuple[ip.ImgArray, ip.ImgArray]:
        random.seed(seed)
            
        subsets: list[np.ndarray] = []
        sum_images = (np.zeros(self.output_shape, dtype=np.float32),
                      np.zeros(self.output_shape, dtype=np.float32))
        next_set = 0
        for subvols in self._iter_chunks(order=order):
            subsets.extend(list(subvols.value))
            next_set = 1 - next_set
            if next_set == 0:
                random.shuffle(subsets)
                lc = len(subsets) // 2
                sum_images[0][:] += sum(subsets[:lc])
                sum_images[1][:] += sum(subsets[lc:])
        
        if next_set == 1:
            random.shuffle(subsets)
            lc = len(subsets) // 2
            sum_images[0][:] += sum(subsets[:lc])
            sum_images[1][:] += sum(subsets[lc:])
            
        random.seed(None)
        img0 = ip.asarray(sum_images[0], axes="zyx")
        img1 = ip.asarray(sum_images[1], axes="zyx")
        img0.set_scale(self.image_ref)
        img1.set_scale(self.image_ref)
        return img0, img1
        
    
    def fsc(
        self,
        mask: ip.ImgArray | None = None,
        seed: int | float | str | bytes | bytearray | None = 0,
        order: int = 1,
        nbin: int = 16,
        ) -> np.ndarray:
        
        # WIP!
        if mask is None:
            mask = 1
        else:
            self._check_shape(mask, "mask")
        
        img0, img1 = self.average_split(seed=seed, order=order)
            
        fsc = ip.fsc(img0*mask,
                     img1*mask,
                     nbin=nbin,
                     r_max=min(self.output_shape)/self.scale/2,
                     )
        
        if self.image_avg is None:
            self.image_avg = img0 + img1
            self.image_avg.set_scale(self.image_ref)
            
        return np.asarray(fsc)
    
    def _check_shape(self, template: ip.ImgArray, name: str = "template") -> None:
        if template.shape != self.output_shape:
            warnings.warn(
                f"'output_shape' of {self.__class__.__name__} object {self.output_shape!r} "
                f"differs from the shape of {name} image {template.shape!r}. 'output_shape' "
                "is updated.",
                UserWarning,
            )
            self.output_shape = template.shape
        return None