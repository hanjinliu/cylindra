from __future__ import annotations
from functools import partial
from typing import Any, Callable, Generator, Generic, Iterator, Iterable, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec
import warnings
import weakref
import tempfile
from scipy.spatial.transform import Rotation
import numpy as np
import impy as ip
from dask import array as da

from ._align_utils import (
    normalize_rotations,
    Ranges,
    align_subvolume,
    align_subvolume_list,
    align_subvolume_multitemplates,
    align_subvolume_list_multitemplates,
    transform_molecules
)
from .molecules import Molecules
from ..utils import multi_map_coordinates, set_gpu
from ..const import nm

if TYPE_CHECKING:
    from ._pca_utils import PcaClassifier
    
_V = TypeVar("_V")
_P = ParamSpec("_P")

class SubtomogramLoader(Generic[_V]):
    """
    A class for efficient loading of subtomograms.
    
    A ``SubtomogramLoader`` instance is basically composed of two elements,
    an image and a Molecules object. A subtomogram is loaded by creating a
    local rotated Cartesian coordinate at a molecule and calculating mapping
    from the image to the subtomogram.
    
    Parameters
    ----------
    image : ImgArray, LazyImgArray, np.ndarray or da.core.Array
        Tomogram image.
    mole : Molecules
        Molecules object that defines position and rotation of subtomograms.
    output_shape : int or tuple of int
        Shape (in pixel) of output subtomograms.
    chunksize : int, optional
        Chunk size used when loading subtomograms.
    """
    
    _PROPS = {
        "ncc": ip.ncc,
        "zncc": ip.zncc,
    }
    
    def __init__(
        self, 
        image: ip.ImgArray | ip.LazyImgArray | np.ndarray | da.core.Array, 
        mole: Molecules,
        output_shape: int | tuple[int, int, int],
        order: int = 1,
        chunksize: int = 1
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
        self._molecules = mole
        self._order = order
        self._chunksize = chunksize
        if isinstance(output_shape, int):
            output_shape = (output_shape,) * ndim
        self._output_shape = output_shape
    
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
        """Get the scale (nm/px) of tomogram."""
        return self.image_ref.scale.x

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    @property
    def molecules(self) -> Molecules:
        return self._molecules

    @property
    def order(self) -> int:
        return self._order
    
    @property
    def chunksize(self) -> int:
        return self._chunksize
    
    def __len__(self) -> int:
        """Return the number of subtomograms."""
        return self.molecules.pos.shape[0]
    
    def __iter__(self) -> Iterator[_V]:
        return self.iter_subtomograms()
    
    def replace(self, order: int | None = None, chunksize: int | None = None):
        if order is None:
            order = self.order
        if chunksize is None:
            chunksize = self.chunksize
        return self.__class__(
            self.image_ref, 
            self.molecules,
            self.output_shape, 
            order=order,
            chunksize=chunksize,
        )
    
    def iter_subtomograms(
        self,
        rotators: Iterable[Rotation] | None = None,
        binsize: int = 1
    ) -> Iterator[ip.ImgArray]:  # axes: zyx or azyx
        if rotators is None:
            iterator = self._iter_chunks()
        else:
            iterator = self._iter_chunks_with_rotation(rotators=rotators)
        
        if binsize == 1:
            for subvols in iterator:
                for subvol in subvols:
                    yield subvol
        else:
            for subvols in iterator:
                for subvol in subvols:  # subvols axes: pazyx
                    subvol: ip.ImgArray  # subvol axes: azyx
                    yield subvol.binning(binsize, check_edges=False)
    
    def iter_subtomograms_with_corr(
        self,
        ref: ip.ImgArray,
        mask: ip.ImgArray | None = None,
        corr_func: Callable[[np.ndarray, np.ndarray], _V] = ip.zncc,
    ) -> Iterator[tuple[ip.ImgArray, _V]]:
        if mask is None:
            mask = 1.0
        ref_masked = ref * mask
        iterator = self._iter_chunks()
        for subvols in iterator:
            for subvol in subvols:
                corr = corr_func(subvol*mask, ref_masked)
                yield subvol, corr
    
    def map(self, f: Callable[_P, _V], *args, **kwargs) -> Iterator[_V]:
        fp = partial(f, *args, **kwargs)
        return map(fp, self.iter_subtomograms)
    
    def iter_to_memmap(self, path: str | None = None):
        shape = (len(self.molecules),) + self.output_shape
        kwargs = dict(dtype=np.float32, mode="w+", shape=shape)
        if path is None:
            with tempfile.NamedTemporaryFile() as ntf:
                mmap = np.memmap(ntf, **kwargs)
        else:
            mmap = np.memmap(path, **kwargs)
            
        for i, subvol in enumerate(self.iter_subtomograms()):
            mmap[i] = subvol
            yield subvol
        darr = da.from_array(mmap, chunks=("auto",) + self.output_shape, meta=np.array([], dtype=np.float32))
        arr = ip.LazyImgArray(darr, name="All_subtomograms", axes="pzyx")
        arr.set_scale(self.image_ref)
        return arr
        
    def to_lazy_imgarray(self, path: str | None = None) -> ip.LazyImgArray:
        it = self.iter_to_memmap(path)
        return self._resolve_iterator(it, lambda x: None)
    
    def to_stack(self, binsize: int = 1) -> ip.ImgArray:
        """Create a 4D image stack of all the subtomograms."""
        images = list(self.iter_subtomograms(binsize=binsize))
        stack: ip.ImgArray = np.stack(images, axis="p")
        stack.set_scale(xyz=self.image_ref.scale.x*binsize)
        return stack

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
        return self.__class__(
            self.image_ref, 
            mole,
            self.output_shape, 
            order=self.order,
            chunksize=self.chunksize,
        )

    def iter_average(
        self,
        classifier: Callable[[np.ndarray], bool] | None = None,
        nbatch: int = 1,
    ) -> Generator[ip.ImgArray, None, ip.ImgArray]:
        aligned = np.zeros(self.output_shape, dtype=np.float32)
        n = 0
        if classifier is None:
            classifier = lambda x: True
        with ip.silent():
            for subvol in self.iter_subtomograms():
                if classifier(subvol):
                    aligned += subvol.value
                n += 1
                if n % nbatch == nbatch - 1:
                    yield aligned
        avg = ip.asarray(aligned / n, name="Avg", axes="zyx")
        avg.set_scale(self.image_ref)
        return avg
    
    def average(
        self,
        *,
        classifier=None, 
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> ip.ImgArray:
        """
        Average all the subtomograms.
        
        The averaged image will be stored in ``self.averaged_image``. The size of subtomograms
        is determined by the ``self.output_shape`` attribute.

        Parameters
        ----------
        classifier : callable, optional
            If given, only those subvolumes that satisfy ``classifier(subvol) == True`` will be
            collected.
        callback : callable, optional
            If given, ``callback(self)`` will be called for each iteration of subtomogram loading.

        Returns
        -------
        ImgArray
            Averaged image.
        """
        average_iter = self.iter_average(classifier=classifier)
        return self._resolve_iterator(average_iter, callback)
    
    def iter_align(
        self,
        *,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.,
        rotations: Ranges | None = None,
        cutoff: float = 0.5,
        nbatch: int = 24,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, SubtomogramLoader]:
        
        if template is None:
            raise NotImplementedError("Template image is needed.")
        if mask is None:
            mask = 1
        self._check_shape(template)
        
        # Convert rotations into quaternion if given.
        rots = normalize_rotations(rotations)
        
        # shift in local Cartesian
        local_shifts = np.zeros((len(self), 3))
        
        # rotation (quaternion) in local Cartesian
        local_rot = np.zeros((len(self), 4))
        local_rot[:, 3] = 1  # identity map in quaternion
        
        _max_shifts_px = np.asarray(max_shifts) / self.scale
        
        with ip.silent(), set_gpu():
            template_masked = template.lowpass_filter(cutoff=cutoff) * mask
            template_ft = template_masked.fft()
        
        if rots is None:
            for i, subvol in enumerate(self.iter_subtomograms()):
                local_shifts[i, :] = align_subvolume(
                    subvol, cutoff, mask, template_ft, _max_shifts_px
                )
                
                if i % nbatch == nbatch - 1:
                    yield local_shifts[i, :], local_rot[i, :]
            
            mole_aligned = self.molecules.translate_internal(local_shifts * self.scale)
                
        else:
            # "rots" is quaternions
            rotators = [Rotation.from_quat(r) for r in rots]
            iterator = self.iter_subtomograms(rotators=rotators)
            for i, subvol_set in enumerate(iterator):
                iopt, local_shifts[i, :] = align_subvolume_list(
                    subvol_set, cutoff, mask, template_ft, template_masked, _max_shifts_px
                )
                local_rot[i, :] = rots[iopt]
                if i % nbatch == nbatch - 1:
                    yield local_shifts[i, :], local_rot[i, :]
            
            mole_aligned = transform_molecules(
                self.molecules, 
                local_shifts* self.scale, 
                Rotation.from_quat(local_rot).as_rotvec()
            )
        
        out = self.__class__(
            self.image_ref,
            mole_aligned, 
            self.output_shape,
            order=self.order,
            chunksize=self.chunksize
        )
        return out
    
    # def iter_align_no_template(
    #     self,
    #     *,
    #     mask_params: np.ndarray | tuple[nm, nm] | Callable[[np.ndarray], np.ndarray] = (1, 1),
    #     max_shifts: nm | tuple[nm, nm, nm] = 1.,
    #     rotations: Ranges | None = None,
    #     corr_threshold = "80%",
    #     cutoff: float = 0.5,
    #     nbatch: int = 24,
    # ) -> Generator[tuple[np.ndarray, np.ndarray], None, SubtomogramLoader]:
                
    #     # Convert rotations into quaternion if given.
    #     rots = normalize_rotations(rotations)
        
    #     # shift in local Cartesian
    #     local_shifts = np.zeros((len(self), 3))
        
    #     # rotation (quaternion) in local Cartesian
    #     local_rot = np.zeros((len(self), 4))
    #     local_rot[:, 3] = 1  # identity map in quaternion
        
    #     _max_shifts_px = np.asarray(max_shifts) / self.scale
    #     all_subvols = self.to_lazy_imgarray(path=None)
        
    #     template = all_subvols.proj("p").compute()
        
    #     with ip.silent(), set_gpu():
    #         template_filt = template.lowpass_filter(cutoff=cutoff)
    #         if isinstance(mask_params, tuple):
    #             _sigma, _radius = mask_params
    #             mask = template.threshold().smooth_mask(
    #                 sigma=_sigma/self.scale, dilate_radius=int(round(_radius/self.scale))
    #             )
    #         elif isinstance(mask_params, np.ndarray):
    #             mask = mask_params
    #         elif callable(mask_params):
    #             mask = mask_params(template)
    #         template_masked = template_filt * mask
    #         # template_ft = template_masked.fft()
        
        
    #     corrs = np.zeros(len(self.molecules))
    #     for i, subvol in enumerate(all_subvols):
    #         corrs[i] = ip.zncc(subvol * mask, template)
        
    #     corr_high = 
        
    #     if rots is None:
    #         for i, subvol in enumerate(self.iter_subtomograms()):
    #             local_shifts[i, :] = align_subvolume(
    #                 subvol, cutoff, mask, template_ft, _max_shifts_px
    #             )
                
    #             if i % nbatch == nbatch - 1:
    #                 yield local_shifts[i, :], local_rot[i, :]
            
    #         mole_aligned = self.molecules.translate_internal(local_shifts * self.scale)
                
    #     else:
    #         # "rots" is quaternions
    #         rotators = [Rotation.from_quat(r) for r in rots]
    #         iterator = self.iter_subtomograms(rotators=rotators)
    #         for i, subvol_set in enumerate(iterator):
    #             iopt, local_shifts[i, :] = align_subvolume_list(
    #                 subvol_set, cutoff, mask, template_ft, template_masked, _max_shifts_px
    #             )
    #             local_rot[i, :] = rots[iopt]
    #             if i % nbatch == nbatch - 1:
    #                 yield local_shifts[i, :], local_rot[i, :]
            
    #         mole_aligned = transform_molecules(
    #             self.molecules, 
    #             local_shifts* self.scale, 
    #             Rotation.from_quat(local_rot).as_rotvec()
    #         )
        
    #     out = self.__class__(
    #         self.image_ref,
    #         mole_aligned, 
    #         self.output_shape,
    #         order=self.order,
    #         chunksize=self.chunksize
    #     )
    #     return out
    
    def iter_align_multi_templates(
        self,
        *,
        templates: list[ip.ImgArray],
        mask: ip.ImgArray | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.,
        rotations: Ranges | None = None,
        cutoff: float = 0.5,
        nbatch: int = 24,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, tuple[np.ndarray, SubtomogramLoader]]:
        
        shapes = [arr.shape for arr in templates]
        if len(set(shapes)) != 1:
            _shapes = ", ".join(str(tuple(shape)) for shape in shapes)
            raise ValueError(f"Inconsistent template shapes: {_shapes}.")
        
        if mask is None:
            mask = 1

        self._check_shape(templates[0])
        
        # Convert rotations into quaternion if given.
        rots = normalize_rotations(rotations)
        
        # optimal shift in local Cartesian
        local_shifts = np.zeros((len(self), 3))
        
        # optimal rotation (quaternion) in local Cartesian
        local_rot = np.zeros((len(self), 4))
        local_rot[:, 3] = 1  # identity map in quaternion
        
        # optimal template ID
        labels = np.zeros(len(self), dtype=np.uint8)
        
        _max_shifts_px = np.asarray(max_shifts) / self.scale
        
        with ip.silent(), set_gpu():
            templates_masked = [tmp.lowpass_filter(cutoff=cutoff) * mask for tmp in templates]
            templates_ft = [tmp.fft() for tmp in templates_masked]

        if rots is None:
            for i, subvol in enumerate(self.iter_subtomograms()):
                iopt, shift = align_subvolume_multitemplates(
                    subvol, cutoff, mask, templates_ft, templates_masked, _max_shifts_px
                )
                local_shifts[i, :] = shift
                labels[i] = iopt
                
                if i % nbatch == nbatch - 1:
                    yield local_shifts[i, :]
            
            mole_aligned = self.molecules.translate_internal(local_shifts * self.scale)
                
        else:
            # "rots" is quaternions
            rotators = [Rotation.from_quat(r) for r in rots]
            iterator = self.iter_subtomograms(rotators=rotators)
            for i, subvol_set in enumerate(iterator):
                iopt, jopt = align_subvolume_list_multitemplates(
                    subvol_set, cutoff, mask, templates_ft, templates_masked, _max_shifts_px
                )
                local_rot[i, :] = rots[iopt]
                labels[i] = jopt
                if i % nbatch == nbatch - 1:
                    yield local_shifts[i, :], local_rot[i, :]
            
            mole_aligned = transform_molecules(
                self.molecules, 
                local_shifts* self.scale, 
                Rotation.from_quat(local_rot).as_rotvec()
            )
        
        out = self.__class__(
            self.image_ref,
            mole_aligned, 
            self.output_shape, 
            order=self.order,
            chunksize=self.chunksize,
        )
        return labels, out
        
    def align(
        self,
        *,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.,
        rotations: Ranges | None = None,
        cutoff: float = 0.5,
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
        rotations : (float, float) or three-tuple of (float, float) or None, optional
            Rotation between subtomograms and template in external Euler angles.
        cutoff : float, default is 0.5
            Cutoff frequency of low-pass filter applied in each subtomogram.
        callback : Callable[[SubtomogramLoader], Any], optional
            Callback function that will get called after each iteration.

        Returns
        -------
        SubtomogramLoader
            Refined molecule object is bound.
        """        
        
        align_iter = self.iter_align(
            template=template, 
            mask=mask,
            max_shifts=max_shifts,
            rotations=rotations,
            cutoff=cutoff
        )
        
        mole_aligned = self._resolve_iterator(align_iter, callback)
        
        out = self.__class__(
            self.image_ref,
            mole_aligned, 
            self.output_shape,
            order=self.order,
            chunksize=self.chunksize,
        )
        
        return out

    def iter_zncc(
        self,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        nbatch: int = 24,
    ) -> Generator[np.ndarray, None, np.ndarray]:
        corrs = np.zeros(len(self.molecules), dtype=np.float32)
        n = 0
        it = self.iter_subtomograms_with_corr(template, mask, corr_func=ip.zncc)
        with ip.silent():
            for i, (subvol, corr) in enumerate(it):
                corrs[i] = corr
                n += 1
                if n % nbatch == nbatch - 1:
                    yield corr

        return corrs
    
    def zncc(
        self,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        callback: Callable[[SubtomogramLoader], Any] = None,
    ):
        align_iter = self.iter_zncc(
            template=template, 
            mask=mask,
        )
        
        return self._resolve_iterator(align_iter, callback)
    
    # @classmethod
    # def register_property(cls, name: str, function: Callable[[np.ndarray, np.ndarray], Any]):
    #     cls._PROPS
        
    def iter_subtomoprops(
        self,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        properties=("zncc",),
        nbatch: int = 24,
    ):
        results = [[]] * len(properties)
        n = 0
        if mask is None:
            mask = 1
        template_masked = template * mask
        with ip.silent():
            for i, subvol in enumerate(self.iter_subtomograms()):
                for _idx_prop, _prop in enumerate(properties):
                    results[_idx_prop].append(_prop(subvol*mask, template_masked))
                n += 1
                if n % nbatch == nbatch - 1:
                    yield

        return results
        
    def iter_each_seam(
        self,
        npf: int,
        template: ip.ImgArray,
        mask: ip.ImgArray | None = None,
    ) -> Generator[tuple[float, ip.ImgArray, np.ndarray],
                   None,
                   tuple[np.ndarray, ip.ImgArray, list[np.ndarray]]]:
        sum_image: np.ndarray = np.zeros((2*npf,) + self.output_shape)
        corrs: list[float] = []
        labels: list[np.ndarray] = []  # list of boolean arrays
        
        if mask is None:
            mask = 1
        
        self._check_shape(template)
        masked_template = template * mask
        _id = np.arange(len(self.molecules))
        
        # prepare all the labels in advance (only takes up ~0.5 MB at most)
        for pf in range(2*npf):
            res = (_id - pf) // npf
            sl = res % 2 == 0
            labels.append(sl)
        
        with ip.silent():
            for idx_chunk, subvols in enumerate(self._iter_chunks()):
                chunk_offset = idx_chunk * self.chunksize
                for j, label in enumerate(labels):
                    sl = label[chunk_offset:chunk_offset + self.chunksize]
                    sum_image[j] += np.sum(subvols[sl], axis=0)
                yield
        
            _n = len(self.molecules) / 2
            avg_image = ip.asarray(sum_image/_n, axes="pzyx")
            corrs = [ip.zncc(avg*mask, masked_template) for avg in avg_image]
        
        avg_image.set_scale(self.image_ref)
        return np.array(corrs), avg_image, labels
    
    def try_all_seams(
        self,
        npf: int,
        template: ip.ImgArray,
        mask: ip.ImgArray | None = None,
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> tuple[np.ndarray, ip.ImgArray, list[Molecules]]:
        
        seam_iter = self.iter_each_seam(
            npf=npf,
            template=template, 
            mask=mask,
        )
        return self._resolve_iterator(seam_iter, callback)
    
    
    def iter_average_split(
        self, 
        *,
        seed: int | float | str | bytes | bytearray | None = 0,
    ):
        np.random.seed(seed)
        try:
            sum0 = np.zeros(self.output_shape, dtype=np.float32)
            sum1 = np.zeros(self.output_shape, dtype=np.float32)
            
            res = 0
            n = 0
            for subvols in self._iter_chunks():
                np.random.shuffle(subvols)
                lc, res = divmod(len(subvols) + res, 2)
                sum0[:] += sum(subvols[:lc])
                sum1[:] += sum(subvols[lc:])
                yield sum0, sum1
        finally:
            np.random.seed(None)
        
        img0 = ip.asarray(sum0, axes="zyx")
        img1 = ip.asarray(sum1, axes="zyx")
        img0.set_scale(self.image_ref)
        img1.set_scale(self.image_ref)
        
        return img0, img1
    
    def average_split(
        self, 
        *,
        seed: int | float | str | bytes | bytearray | None = 0,
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> tuple[ip.ImgArray, ip.ImgArray]:
        it = self.iter_average_split(seed=seed)
        return self._resolve_iterator(it, callback)
        
    
    def fsc(
        self,
        mask: ip.ImgArray | None = None,
        seed: int | float | str | bytes | bytearray | None = 0,
        dfreq: float = 0.05,
        ) -> tuple[np.ndarray, np.ndarray]:
        
        if mask is None:
            mask = 1
        else:
            self._check_shape(mask, "mask")
        
        img0, img1 = self.average_split(seed=seed)
        
        with ip.silent():
            freq, fsc = ip.fsc(img0*mask, img1*mask, dfreq=dfreq)
        
        return freq, fsc
    
    def get_classifier(
        self,
        mask: ip.ImgArray | None = None,
        n_components: int = 5,
        n_clusters: int = 2,
        binsize: int = 1,
        seed: int | None = 0,
    ) -> PcaClassifier:
        image_stack = self.to_stack(binsize=binsize)
        from ._pca_utils import PcaClassifier
        clf = PcaClassifier(
            image_stack,
            mask_image=mask, 
            n_components=n_components,
            n_clusters=n_clusters,
            seed=seed,
        )
        return clf
    
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
    
    def _iter_chunks(self) -> Iterator[ip.ImgArray]:  # axes: pzyx
        """Generate subtomogram list chunk-wise."""
        image = self.image_ref
        scale = image.scale.x
        
        with ip.silent():
            for coords in self.molecules.iter_cartesian(self.output_shape, scale, self.chunksize):
                with set_gpu():
                    subvols = np.stack(
                        multi_map_coordinates(image, coords, order=self.order, cval=np.mean),
                        axis=0,
                    )
                subvols = ip.asarray(subvols, axes="pzyx")
                subvols.set_scale(image)
                yield subvols
    
    def _iter_chunks_with_rotation(
        self, 
        rotators: list[Rotation],
    ) -> Iterator[ip.ImgArray]:  # axes: pazyx
        image = self.image_ref
        matrices = _compose_rotation_matrices(self.output_shape, rotators)
        for subvols in self._iter_chunks():
            # shape: (position, rot-angle, z, y, x)
            with ip.silent(), set_gpu():
                all_subvols: list[ip.ImgArray] = []  # axes: pzyx
                for mtx in matrices:
                    all_subvols.append(subvols.affine(mtx, dims="zyx").value)
                out = np.stack(all_subvols, axis=1)
                out = ip.asarray(out, axes="pazyx")
                out.set_scale(image)
            yield out
    
        
    def _resolve_iterator(self, it: Generator[Any, Any, _V], callback: Callable) -> _V:
        """Iterate over an iterator until it returns something."""
        if callback is None:
            callback = lambda x: None
        while True:
            try:
                next(it)
                callback(self)
            except StopIteration as e:
                results = e.value
                break
        
        return results

def _compose_rotation_matrices(
    shape: tuple[int, int, int],
    rotators: list[Rotation],
):
    dz, dy, dx = (np.array(shape) - 1) / 2
    # center to corner
    translation_0 = np.array([[1., 0., 0., dz],
                                [0., 1., 0., dy],
                                [0., 0., 1., dx],
                                [0., 0., 0., 1.]],
                                dtype=np.float32)
    # corner to center
    translation_1 = np.array([[1., 0., 0., -dz],
                                [0., 1., 0., -dy],
                                [0., 0., 1., -dx],
                                [0., 0., 0.,  1.]],
                                dtype=np.float32)
    
    matrices = []
    for rot in rotators:
        e_ = np.eye(4)
        e_[:3, :3] = rot.as_matrix()
        matrices.append(translation_0 @ e_ @ translation_1)
    return matrices
