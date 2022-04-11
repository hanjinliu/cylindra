from __future__ import annotations
from functools import partial
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterator,
    TYPE_CHECKING,
    TypeVar
)
from typing_extensions import ParamSpec
import warnings
import weakref
import tempfile
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy import ndimage as ndi
import numpy as np
import impy as ip
from dask import array as da
from .align import (
    Ranges,
    transform_molecules,
    AlignmentModel,
    AlignmentResult,
)
from .molecules import Molecules
from ..utils import multi_map_coordinates, set_gpu
from ..const import nm, Mole, Align

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
    order : int, default is 1
        Interpolation order of subtomogram sampling.
        - 0 = Nearest neighbor
        - 1 = Linear interpolation
        - 3 = Cubic interpolation
    chunksize : int, optional
        Chunk size used when loading subtomograms. This parameter controls the
        number of subtomograms to be loaded at the same time. Larger chunk size
        results in better performance if adjacent subtomograms are near to each
        other.
    """
    
    def __init__(
        self, 
        image: ip.ImgArray | ip.LazyImgArray | np.ndarray | da.core.Array, 
        mole: Molecules,
        output_shape: int | tuple[int, int, int],
        order: int = 1,
        chunksize: int = 1,
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
        self._chunksize = max(chunksize, 1)
        if isinstance(output_shape, int):
            output_shape = (output_shape,) * ndim
        else:
            output_shape = tuple(output_shape)
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
        """Return the output subtomogram shape."""
        return self._output_shape

    @property
    def molecules(self) -> Molecules:
        """Return the molecules of the subtomogram loader."""
        return self._molecules

    @property
    def order(self) -> int:
        """Return the interpolation order."""
        return self._order
    
    @property
    def chunksize(self) -> int:
        """Return the chunk size on subtomogram loading."""
        return self._chunksize

    def __len__(self) -> int:
        """Return the number of subtomograms."""
        return self.molecules.pos.shape[0]
    
    def __iter__(self) -> Iterator[_V]:
        return self.iter_subtomograms()
    
    def replace(
        self,
        output_shape: int | tuple[int, int, int] | None = None,
        order: int | None = None,
        chunksize: int | None = None
    ):
        """Return a new instance with different parameter(s)."""
        if output_shape is None:
            output_shape = self.output_shape
        if order is None:
            order = self.order
        if chunksize is None:
            chunksize = self.chunksize
        return self.__class__(
            self.image_ref, 
            self.molecules,
            output_shape=output_shape, 
            order=order,
            chunksize=chunksize,
        )
    
    def iter_subtomograms(
        self,
        binsize: int = 1
    ) -> Iterator[ip.ImgArray]:  # axes: zyx
        """
        Iteratively load subtomograms.
        
        This method load the required region of tomogram into memory and crop
        subtomograms from it. To avoid repetitively loading same region, this
        function determines what range of tomogram is needed to load current
        chunk of subtomograms.

        Parameters
        ----------
        binsize : int, default is 1
            Image bin size.

        Yields
        ------
        ip.ImgArray
            Yields image array of each subtomogram.
        """
        if binsize == 1:
            for subvols in self._iter_chunks():
                for subvol in subvols:
                    yield subvol
        else:
            for subvols in self._iter_chunks():
                for subvol in subvols:  # subvols axes: pzyx
                    subvol: ip.ImgArray  # subvol axes: zyx
                    yield subvol.binning(binsize, check_edges=False)
    
    def map(self, f: Callable[_P, _V], *args, **kwargs) -> Iterator[_V]:
        """
        Map subtomogram loader to a function.
        
        For every subtomogram ``img`` this function yields
        ``f(img, *args, **kwargs)``.

        Parameters
        ----------
        f : callable
            Any mapping function.

        """
        fp = partial(f, *args, **kwargs)
        return map(fp, self.iter_subtomograms)
    
    def iter_to_memmap(self, path: str | None = None):
        """
        Create an iterator that convert all the subtomograms into a memory-mapped array.
        
        This function is useful when the same set of subtomograms will be used for many
        times but it should not be fully loaded into memory. A temporary file will be
        created to store subtomograms by default.

        Parameters
        ----------
        path : str, optional
            File path of the temporary file. If not given file will be created  by
            ``tempfile.NamedTemporaryFile`` function.

        Returns
        -------
        LazyImgArray
            A lazy-loading array that uses the memory-mapped array.

        Yields
        ------
        ImgArray
            Subtomogram at each position.
        """
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
        darr = da.from_array(mmap, chunks=(1,) + self.output_shape, meta=np.array([], dtype=np.float32))
        arr = ip.LazyImgArray(darr, name="All_subtomograms", axes="pzyx")
        arr.set_scale(self.image_ref)
        return arr
        
    def to_lazy_imgarray(self, path: str | None = None) -> ip.LazyImgArray:
        """
        An non-iterator version of :func:`iter_to_memmap`.

        Parameters
        ----------
        path : str, optional
            File path of the temporary file. If not given file will be created  by
            ``tempfile.NamedTemporaryFile`` function.

        Returns
        -------
        LazyImgArray
            A lazy-loading array that uses the memory-mapped array.
        
        Examples
        --------
        1. Get i-th subtomogram.
        
        >>> arr = loader.to_lazy_imgarray()  # axes = "pzyx"
        >>> arr[i]
        
        2. Subtomogram averaging.
        
        >>> arr = loader.to_lazy_imgarray()  # axes = "pzyx"
        >>> avg = arr.proj("p")  # identical to np.mean(arr, axis=0)
        
        """
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
    ) -> Generator[ip.ImgArray, None, ip.ImgArray]:
        """
        Create an iterator that calculate the averaged image from a tomogram.
        
        This function execute so-called "subtomogram averaging". The size of 
        subtomograms is determined by the ``self.output_shape`` attribute.

        Parameters
        ----------
        classifier : callable, optional
            If provided, only subtomograms that satisfy ``classifier(img)==True`` will
            be used.

        Returns
        -------
        ImgArray
            Averaged image

        Yields
        ------
        ImgArray
            Subtomogram at each position
        """
        aligned = np.zeros(self.output_shape, dtype=np.float32)
        n = 0
        if classifier is None:
            classifier = lambda x: True
        with ip.silent():
            for subvol in self.iter_subtomograms():
                if classifier(subvol):
                    aligned += subvol.value
                n += 1
                yield aligned
        avg = ip.asarray(aligned / n, name="Avg", axes="zyx")
        avg.set_scale(self.image_ref)
        return avg
    
    def iter_align(
        self,
        template: ip.ImgArray,
        *,
        mask: ip.ImgArray = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.,
        rotations: Ranges | None = None,
        cutoff: float = 0.5,
        method: str = "pcc",
    ) -> Generator[AlignmentResult, None, SubtomogramLoader]:
        """
        Create an iterator that align subtomograms to the template image.
        
        This method conduct so called "subtomogram refinement". Only shifts and rotations
        are calculated in this method. To get averaged image, you'll have to run "average"
        method using the resulting SubtomogramLoader instance.
        
        Parameters
        ----------
        template : ip.ImgArray, optional
            Template image.
        mask : ip.ImgArray, optional
            Mask image. Must in the same shae as the template.
        max_shifts : int or tuple of int, default is (1., 1., 1.)
            Maximum shift between subtomograms and template.
        rotations : (float, float) or three-tuple of (float, float) or None, optional
            Rotation between subtomograms and template in external Euler angles.
        cutoff : float, default is 0.5
            Cutoff frequency of low-pass filter applied in each subtomogram.
        method : {"pcc", "zncc"}, default is "pcc"
            Alignment method. "pcc": phase cross correlation; "zncc": zero-mean normalized
            cross correlation.

        Returns
        -------
        SubtomogramLoader
            An loader instance with updated molecules.

        Yields
        ------
        AlignmentResult
            An tuple representing the current alignment result.
        """

        self._check_shape(template)
        
        local_shifts, local_rot, corr_max = _allocate(len(self))
        _max_shifts_px = np.asarray(max_shifts) / self.scale
        
        with ip.silent(), set_gpu():
            model = AlignmentModel(
                template=template, 
                mask=mask, 
                cutoff=cutoff, 
                rotations=rotations,
                method=method
            )
            for i, subvol in enumerate(self.iter_subtomograms()):
                result = model.align(subvol, _max_shifts_px)
                _, local_shifts[i], local_rot[i], corr_max[i] = result
                yield result
            
            rotvec = Rotation.from_quat(local_rot).as_rotvec()
            mole_aligned = transform_molecules(
                self.molecules, 
                local_shifts * self.scale, 
                rotvec,
            )
        
        mole_aligned.features = pd.concat(
            [self.molecules.features,
             get_features(method, corr_max, local_shifts, rotvec)],
            axis=1)

        out = self.__class__(
            self.image_ref,
            mole_aligned, 
            self.output_shape,
            order=self.order,
            chunksize=self.chunksize,
        )
        
        return out
    
    def iter_align_no_template(
        self,
        *,
        mask_params: np.ndarray | tuple[nm, nm] | Callable[[np.ndarray], np.ndarray] = (1, 1),
        max_shifts: nm | tuple[nm, nm, nm] = 1.,
        rotations: Ranges | None = None,
        cutoff: float = 0.5,
        method: str = "pcc",
    ) -> Generator[AlignmentResult, None, SubtomogramLoader]:
        """
        Create an iterator that align subtomograms without template image.
        
        A template-free version of :func:`iter_align`. This method first calculates averaged
        image and uses it for the alignment template. To avoid loading same subtomograms
        twice, a memory-mapped array is created internally (so the second subtomogram 
        loading is faster).
        
        Parameters
        ----------
        template : ip.ImgArray, optional
            Template image.
        mask : ip.ImgArray, optional
            Mask image. Must in the same shae as the template.
        max_shifts : int or tuple of int, default is (1., 1., 1.)
            Maximum shift between subtomograms and template.
        rotations : (float, float) or three-tuple of (float, float) or None, optional
            Rotation between subtomograms and template in external Euler angles.
        cutoff : float, default is 0.5
            Cutoff frequency of low-pass filter applied in each subtomogram.
        method : {"pcc", "zncc"}, default is "pcc"
            Alignment method. "pcc": phase cross correlation; "zncc": zero-mean normalized
            cross correlation.

        Returns
        -------
        SubtomogramLoader
            An loader instance with updated molecules.

        Yields
        ------
        AlignmentResult
            An tuple representing the current alignment result.
        """
        local_shifts, local_rot, corr_max = _allocate(len(self))
        _max_shifts_px = np.asarray(max_shifts) / self.scale
        all_subvols = yield from self.iter_to_memmap(path=None)
        
        with ip.silent():
            template = all_subvols.proj("p").compute()
        
        # get mask image
        if isinstance(mask_params, tuple):
            _sigma, _radius = mask_params
            with ip.silent():
                mask = template.threshold().smooth_mask(
                    sigma=_sigma/self.scale,
                    dilate_radius=int(round(_radius/self.scale)),
                )
        elif isinstance(mask_params, np.ndarray):
            mask = mask_params
        elif callable(mask_params):
            mask = mask_params(template)
        else:
            mask = mask_params
            
        
        with ip.silent(), set_gpu():
            model = AlignmentModel(
                template=template,
                mask=mask,
                cutoff=cutoff,
                rotations=rotations,
                method=method,
            )
            for i, subvol in enumerate(all_subvols):
                result = model.align(subvol.compute(), _max_shifts_px)
                _, local_shifts[i], local_rot[i], corr_max[i] = result
                yield result
            
            rotvec = Rotation.from_quat(local_rot).as_rotvec()
            mole_aligned = transform_molecules(
                self.molecules, 
                local_shifts * self.scale, 
                rotvec,
            )
        
        mole_aligned.features = pd.concat(
            [self.molecules.features,
             get_features(method, corr_max, local_shifts, rotvec)],
            axis=1,
        )
        
        out = self.__class__(
            self.image_ref,
            mole_aligned, 
            self.output_shape,
            order=self.order,
            chunksize=self.chunksize,
        )
        return out
    
    def iter_align_multi_templates(
        self,
        *,
        templates: list[ip.ImgArray],
        mask: ip.ImgArray | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.,
        rotations: Ranges | None = None,
        cutoff: float = 0.5,
        method: str = "pcc",
    ) -> Generator[AlignmentResult, None, SubtomogramLoader]:
        n_templates = len(templates)
        self._check_shape(templates[0])
        
        local_shifts, local_rot, corr_max = _allocate(len(self))
        
        # optimal template ID
        labels = np.zeros(len(self), dtype=np.uint32)
        
        _max_shifts_px = np.asarray(max_shifts) / self.scale
        with ip.silent(), set_gpu():
            model = AlignmentModel(
                template=np.stack(list(templates), axis="p"),
                mask=mask,
                cutoff=cutoff,
                rotations=rotations,
                method=method,
            )

            for i, subvol in enumerate(self.iter_subtomograms()):
                result = model.align(subvol, _max_shifts_px)
                labels[i], local_shifts[i], local_rot[i], corr_max[i] = result
                yield result
            rotvec = Rotation.from_quat(local_rot).as_rotvec()
            mole_aligned = transform_molecules(
                self.molecules, 
                local_shifts* self.scale, 
                Rotation.from_quat(local_rot).as_rotvec()
            )
        
        if model.has_rotation:
            labels %= n_templates
        labels = labels.astype(np.uint8)
        
        mole_aligned.features = pd.concat(
            [self.molecules.features,
             get_features(method, corr_max, local_shifts, rotvec),
             pd.DataFrame({"labels": labels})],
            axis=1,
        )
        
        out = self.__class__(
            self.image_ref,
            mole_aligned, 
            self.output_shape, 
            order=self.order,
            chunksize=self.chunksize,
        )
        return out
    
    
    def iter_subtomoprops(
        self,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        properties=(ip.zncc,),
    ) -> Generator[None, None, pd.DataFrame]:
        results = {f.__name__: np.zeros(len(self), dtype=np.float32) for f in properties}
        n = 0
        if mask is None:
            mask = 1
        template_masked = template * mask
        with ip.silent():
            for i, subvol in enumerate(self.iter_subtomograms()):
                for _prop in properties:
                    prop = _prop(subvol*mask, template_masked)
                    results[_prop.__name__][i] = prop
                n += 1
                yield
        
        return pd.DataFrame(results)


    # def iter_constrained_align(
    #     self,
    #     *,
    #     template: ip.ImgArray = None,
    #     mask: ip.ImgArray = None,
    #     max_shifts: nm | tuple[nm, nm, nm] = 1.,
    #     rotations: Ranges | None = None,
    #     cutoff: float = 0.5,
    #     method: str = "pcc",
    #     npf: int = 13,
    # ) -> Generator[AlignmentResult, None, SubtomogramLoader]:
    #     from scipy.optimize import dual_annealing
        
    #     if template is None:
    #         raise NotImplementedError("Template image is needed.")
        
    #     self._check_shape(template)
        
    #     local_shifts, local_rot, corr_max = _allocate(len(self))
    #     _max_shifts_px = np.asarray(max_shifts) / self.scale
    #     _landscapes: list[np.ndarray] = []
        
    #     with ip.silent(), set_gpu():
    #         model = AlignmentModel(
    #             template=template, 
    #             mask=mask, 
    #             cutoff=cutoff, 
    #             rotations=rotations,
    #             method=method
    #         )
    #         for i, subvol in enumerate(self.iter_subtomograms()):
    #             lds = model.landscape(subvol, _max_shifts_px)
    #             _landscapes.append(lds.value)
    #             yield lds
        
    #     _max_shifts_ceil = np.ceil(_max_shifts_px)
    #     lds_shape = 1 + 2*_max_shifts_ceil
    #     landscapes = np.stack(_landscapes, axis=0).reshape(-1, npf, *lds_shape)
    #     bounds = np.concatenate(
    #         [np.stack(
    #             [_max_shifts_ceil - _max_shifts_px, 
    #              _max_shifts_ceil + _max_shifts_px],
    #             axis=1)
    #          ]*len(self),
    #         axis=0,
    #     )

    #     results = dual_annealing(
    #         _neg_correlation_total,
    #         bounds=bounds,
    #         args=(landscapes,)
    #     )
        
    #     local_shifts = results.x.reshape(-1, 3) - _max_shifts_ceil[np.newaxis, :]
    #     rotvec = np.zeros((len(self), 3))
    #     mole_aligned = transform_molecules(
    #         self.molecules, 
    #         local_shifts * self.scale, 
    #         rotvec,
    #     )
        
    #     mole_aligned.features = pd.concat(
    #         [self.molecules.features,
    #          get_features(method, corr_max, local_shifts, rotvec)],
    #         axis=1)

    #     out = self.__class__(
    #         self.image_ref,
    #         mole_aligned, 
    #         self.output_shape,
    #         order=self.order,
    #         chunksize=self.chunksize,
    #     )
        
    #     return out
    
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
    
    
    def iter_average_split(
        self, 
        *,
        n_set: int = 1,
        seed: int | float | str | bytes | bytearray | None = 0,
    ):
        np.random.seed(seed)
        try:
            sum_images = np.zeros((n_set, 2) + self.output_shape, dtype=np.float32)
            res = 0
            for subvols in self._iter_chunks():
                lc, res = divmod(len(subvols) + res, 2)
                for i_set in range(n_set):
                    np.random.shuffle(subvols)
                    sum_images[i_set, 0] += np.sum(subvols[:lc], axis=0)
                    sum_images[i_set, 1] += np.sum(subvols[lc:], axis=0)
                yield sum_images
        finally:
            np.random.seed(None)
        
        img = ip.asarray(sum_images, axes="pqzyx")
        img.set_scale(self.image_ref)
        
        return img
    
    
    def average(
        self,
        *,
        classifier=None, 
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> ip.ImgArray:
        """
        A non-iterator version of :func:`iter_average`.

        Parameters
        ----------
        classifier : callable, optional
            If given, only those subvolumes that satisfy ``classifier(subvol) == True`` 
            will be collected.
        callback : callable, optional
            If given, ``callback(self)`` will be called for each iteration of subtomogram 
            loading.

        Returns
        -------
        ImgArray
            Averaged image.
        """
        average_iter = self.iter_average(classifier=classifier)
        return self._resolve_iterator(average_iter, callback)
        
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
        A non-iterator version of :func:`iter_align`.
        
        Parameters
        ----------
        template : ip.ImgArray, optional
            Template image.
        mask : ip.ImgArray, optional
            Mask image. Must in the same shae as the template.
        max_shifts : int or tuple of int, default is (1., 1., 1.)
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

    def average_split(
        self, 
        *,
        n_set: int = 1,
        seed: int | float | str | bytes | bytearray | None = 0,
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> tuple[ip.ImgArray, ip.ImgArray]:
        it = self.iter_average_split(n_set=n_set, seed=seed)
        return self._resolve_iterator(it, callback)
        
    
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
    
    
    def fsc(
        self,
        mask: ip.ImgArray | None = None,
        seed: int | float | str | bytes | bytearray | None = 0,
        n_set: int = 1,
        dfreq: float = 0.05,
        ) -> pd.DataFrame:
        
        if mask is None:
            mask = 1
        else:
            self._check_shape(mask, "mask")
        
        img = self.average_split(n_set=n_set, seed=seed)
        fsc_all: dict[str, np.ndarray] = {}
        with ip.silent():
            for i in range(n_set):
                img0, img1 = img[i]
                freq, fsc = ip.fsc(img0*mask, img1*mask, dfreq=dfreq)
                fsc_all[f"FSC-{i}"] = fsc
        
        df = pd.DataFrame({"freq": freq})
        return df.update(fsc_all)
    
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


def get_features(method: str, corr_max, local_shifts, rotvec) -> pd.DataFrame:
    feature_key = Mole.zncc if method == "zncc" else Mole.pcc
    
    features = {
        feature_key: corr_max,
        Align.zShift: local_shifts[:, 0],
        Align.yShift: local_shifts[:, 1],
        Align.xShift: local_shifts[:, 2],
        Align.zRotvec: rotvec[:, 0],
        Align.yRotvec: rotvec[:, 1],
        Align.xRotvec: rotvec[:, 2],
    }
    return pd.DataFrame(features)


def _allocate(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # shift in local Cartesian
    local_shifts = np.zeros((size, 3))
    
    # maximum ZNCC
    corr_max = np.zeros(size)
    
    # rotation (quaternion) in local Cartesian
    local_rot = np.zeros((size, 4))
    local_rot[:, 3] = 1  # identity map in quaternion
    
    return local_shifts, local_rot, corr_max

# def _neg_correlation_total(x: np.ndarray, lds: np.ndarray):
#     # x[3*i], x[3*i + 1], x[3*i + 2] is (z, y, x) coordinates in i-th landscape
#     coords = x.reshape(-1, 3)
#     return -np.sum(
#         ndi.map_coordinates(lds, coords.T, order=3, prefilter=True)
#     )