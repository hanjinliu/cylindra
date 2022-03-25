from __future__ import annotations
from typing import Iterable, Iterator
import warnings
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from ..const import EulerAxes, nm

_CSV_COLUMNS = ["z", "y", "x", "zvec", "yvec", "xvec"]

class Molecules:
    """
    Object that represents multiple orientation and position of molecules. Orientation
    is represented by `scipy.spatial.transform.Rotation`. **All the vectors are zyx-order**.
    """
    def __init__(self, pos: np.ndarray, rot: Rotation):
        pos = np.atleast_2d(pos)
        
        if pos.shape[1] != 3:
            raise ValueError("Shape of pos must be (N, 3).")
        elif pos.shape[0] != len(rot):
            raise ValueError(
                f"Length mismatch. There are {pos.shape[0]} molecules but {len(rot)} "
                "rotation were given."
                )
        
        self._pos = pos
        self._rotator = rot
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={len(self)})"
    
    @classmethod
    def from_axes(
        cls, 
        pos: np.ndarray, 
        z: np.ndarray | None = None, 
        y: np.ndarray | None = None,
        x: np.ndarray | None = None,
    ) -> Molecules:
        """Construct molecule cloud with orientation from two of their local axes."""
        pos = np.atleast_2d(pos)
        
        if sum((_ax is not None) for _ax in [z, y, x]) != 2:
            raise TypeError("You must specify two out of z, y, and x.")
        
        # NOTE: np.cross assumes vectors are in xyz order. However, all the arrays here are defined
        # in zyx order. To build right-handed coordinates, we must invert signs when using np.cross.
        if z is None:
            x = np.atleast_2d(x)
            y = np.atleast_2d(y)
            z = -np.cross(x, y, axis=1)
        elif y is None:
            z = np.atleast_2d(z)
            x = np.atleast_2d(x)
            y = -np.cross(z, x, axis=1)
        
        rotator = axes_to_rotator(z, y)
        return cls(pos, rotator)

    @classmethod
    def from_euler(
        cls,
        pos: np.ndarray,
        angles: np.ndarray, 
        seq: str | EulerAxes = EulerAxes.ZXZ,
        degrees: bool = False
    ):
        """Create molecules from Euler angles."""
        rotator = from_euler(angles, seq, degrees)
        return cls(pos, rotator)
    
    @classmethod
    def from_csv(cls, path: str, **kwargs) -> Molecules:
        """Load csv as a Molecules object."""
        df: pd.DataFrame = pd.read_csv(path, **kwargs)
        if df.shape[1] < 6:
            raise ValueError(f"CSV must have more than or equal six columns but got shape {df.shape}")
        if list(df.columns)[:6] != _CSV_COLUMNS:
            warnings.warn(
                f"Columns {df.columns} does not match the standard column names for Molecules object.",
                UserWarning,
            )
        
        return cls(df.values[:, :3], Rotation.from_rotvec(df.values[:, 3:6]))
    
    def to_dataframe(self) -> pd.DataFrame:
        rotvec = self.rotvec()
        data = np.concatenate([self.pos, rotvec], axis=1)
        df = pd.DataFrame(data, columns=_CSV_COLUMNS)
        return df

    def to_csv(self, save_path: str, properties: pd.DataFrame | None = None) -> None:
        """
        Save molecules as a csv file.

        Parameters
        ----------
        save_path : str
            Save path.
        properties : pd.DataFrame | None, optional
            Molecules porperties. Saved in the same csv file from column-7.
        """        
        df = self.to_dataframe()
        if properties is not None:
            if len(properties) != len(self):
                raise ValueError("Length mismatch between Molecules and properties.")
            df = pd.concat([df, properties], axis=1)
        df.to_csv(save_path, index=False)
        return None

    def __len__(self) -> int:
        """Return the number of molecules."""
        return self._pos.shape[0]
    
    def __getitem__(self, key: int | slice | list[int] | np.ndarray) -> Molecules:
        return self.subset(key)
    
    @property
    def pos(self) -> np.ndarray:
        """Positions of molecules."""
        return self._pos
    
    @property
    def x(self) -> np.ndarray:
        """Vectors of x-axis."""
        return self._rotator.apply([0., 0., 1.])
    
    @property
    def y(self) -> np.ndarray:
        """Vectors of y-axis."""
        return self._rotator.apply([0., 1., 0.])
    
    @property
    def z(self) -> np.ndarray:
        """Vectors of z-axis."""
        return self._rotator.apply([1., 0., 0.])

    @property
    def rotator(self) -> Rotation:
        return self._rotator
    
    @classmethod
    def concat(cls, moles: Iterable[Molecules]) -> Molecules:
        """Concatenate Molecules objects."""
        pos: list[np.ndarray] = []
        quat: list[np.ndarray] = []
        for mol in moles:
            pos.append(mol.pos)
            quat.append(mol.quaternion())
        
        all_pos = np.concatenate(pos, axis=0)
        all_quat = np.concatenate(quat, axis=0)
        
        return cls(all_pos, Rotation(all_quat))
    
    def subset(self, spec: int | slice | list[int] | np.ndarray) -> Molecules:
        """
        Create a subset of molecules by slicing.
        
        Any slicing supported in ``numpy.ndarray``, except for integer, can be used here.
        Molecule positions and angles are sliced at the same time.

        Parameters
        ----------
        spec : int, slice, list of int, or ndarray
            Specifier that defines which molecule will be used. Any objects that numpy
            slicing are defined are supported. For instance, ``[2, 3, 5]`` means the 2nd,
            3rd and 5th molecules will be used (zero-indexed), and ``slice(10, 20)``
            means the 10th to 19th molecules will be used.

        Returns
        -------
        Molecules
            Molecule subset.
        """
        if isinstance(spec, int):
            spec = slice(spec, spec+1)
        pos = self.pos[spec]
        quat = self._rotator.as_quat()[spec]
        return self.__class__(pos, Rotation(quat))

    def affine_matrix(self, center):
        nmole = len(self)
        mat = self.matrix()
        rot_mat = np.zeros((nmole, 4, 4), dtype=np.float32)
        rot_mat[:, 3, 3] = 1.
        rot_mat[:, :3, :3] = mat
        
        translation_0 = np.stack([np.eye(4, dtype=np.float32)] * nmole, axis=0)
        translation_1 = np.stack([np.eye(4, dtype=np.float32)] * nmole, axis=0)
        translation_0[:, :3, 3] = center
        translation_1[:, :3, 3] = -center
        
        return np.einsum("nij,njk,nkl->nil", translation_0, rot_mat, translation_1)
        

    def cartesian(self, shape: tuple[int, int, int], scale: nm) -> np.ndarray:
        """
        Return all the rotated Cartesian coordinate systems defined around each molecule.
        
        If number of molecules is very large, this method could raise memory error. To avoid it,
        ``iter_cartesian`` could be an alternative.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Shape of output coordinates.
        scale : nm
            Scale of coordinates. Same as the pixel size of the tomogram.

        Yields
        ------
        np.ndarray
            An array of shape (N, D, Lz, Ly, Lx), where N is the chunk size, D is
            the number of dimensions (=3) and rests are length along each dimension.
        """
        it = self.iter_cartesian(shape, scale, chunksize=len(self))
        return next(it)
    
    def cartesian_dask(
        self, 
        shape: tuple[int, int, int], 
        scale: nm,
        chunksize: int = 64,
    ):
        center = np.array(shape) / 2 - 0.5
        vec_x = self.x
        vec_y = self.y
        vec_z = -np.cross(vec_x, vec_y, axis=1)
        ind_z, ind_y, ind_x = [np.arange(s) - c for s, c in zip(shape, center)]
        
        x_ax = vec_x[:, :, np.newaxis] * ind_x
        y_ax = vec_y[:, :, np.newaxis] * ind_y
        z_ax = vec_z[:, :, np.newaxis] * ind_z
        
        # There will be many points so data type should be converted into 32-bit
        x_ax = x_ax.astype(np.float32)
        y_ax = y_ax.astype(np.float32)
        z_ax = z_ax.astype(np.float32)
        
        from dask import array as da
        x_ax = da.from_array(x_ax, chunks=(chunksize,) + x_ax.shape[1:])
        x_ax = da.from_array(y_ax, chunks=(chunksize,) + y_ax.shape[1:])
        x_ax = da.from_array(z_ax, chunks=(chunksize,) + z_ax.shape[1:])
        
        coords = (
            z_ax[:, :, :, np.newaxis, np.newaxis]
            + y_ax[:, :, np.newaxis, :, np.newaxis] 
            + x_ax[:, :, np.newaxis, np.newaxis, :]
            )
        shifts = self.pos / scale
        coords += shifts[:, :, np.newaxis, np.newaxis, np.newaxis]  # unit: pixel
        return coords
        
    def iter_cartesian(
        self, 
        shape: tuple[int, int, int], 
        scale: nm,
        chunksize: int = 100,
    ) -> Iterator[np.ndarray]:
        """
        Iterate over all the rotated Cartesian coordinate systems defined around each molecule.
        
        Coordinates are split by ``chunksize`` because this method usually creates very large
        arrays but they will not be used at the same time. To get whole list of coordinates, use
        ``cartesian`` method.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Shape of output coordinates.
        scale : nm
            Scale of coordinates. Same as the pixel size of the tomogram. This parameter is 
            needed for calculating the center pixel coordinates.
        chunksize : int, default is 100
            Chunk size of the iterator.

        Yields
        ------
        Iterator[np.ndarray]
            Every yield is an array of shape (N, D, Lz, Ly, Lx), where N is the chunk size, D is
            the number of dimensions (=3) and rests are length along each dimension.
        """
        center = np.array(shape) / 2 - 0.5
        vec_x = self.x
        vec_y = self.y
        vec_z = -np.cross(vec_x, vec_y, axis=1)
        ind_z, ind_y, ind_x = [np.arange(s) - c for s, c in zip(shape, center)]
        
        chunk_offset = 0
        nmole = len(self)
        while chunk_offset < nmole:
            sl = slice(chunk_offset, chunk_offset + chunksize, None)
            x_ax = vec_x[sl, :, np.newaxis] * ind_x
            y_ax = vec_y[sl, :, np.newaxis] * ind_y
            z_ax = vec_z[sl, :, np.newaxis] * ind_z
            
            # There will be many points so data type should be converted into 32-bit
            x_ax = x_ax.astype(np.float32)
            y_ax = y_ax.astype(np.float32)
            z_ax = z_ax.astype(np.float32)
            
            coords = (
                z_ax[:, :, :, np.newaxis, np.newaxis]
                + y_ax[:, :, np.newaxis, :, np.newaxis] 
                + x_ax[:, :, np.newaxis, np.newaxis, :]
                )
            shifts = self.pos[sl] / scale
            coords += shifts[:, :, np.newaxis, np.newaxis, np.newaxis]  # unit: pixel
            yield coords
            chunk_offset += chunksize
        
    def matrix(self) -> np.ndarray:
        """
        Calculate rotation matrices that align molecules in such orientations that ``vec``
        belong to the object.

        Returns
        -------
        (N, 3, 3) ndarray
            Rotation matrices. Rotations represented by these matrices transform molecules
            to the same orientations, i.e., align all the molecules.
        """
        return self._rotator.as_matrix()
    
    
    def euler_angle(self, 
                    seq: str | EulerAxes = EulerAxes.ZXZ,
                    degrees: bool = False) -> np.ndarray:
        """
        Calculate Euler angles that transforms a source vector to vectors that 
        belong to the object.

        Parameters
        ----------
        seq : str, default is "ZXZ"
            Copy of ``scipy.spatial.transform.Rotation.as_euler``. 3 characters 
            belonging to the set {"X", "Y", "Z"} for intrinsic rotations, or
            {"x", "y", "z"} for extrinsic rotations. Adjacent axes cannot be the 
            same. Extrinsic and intrinsic rotations cannot be mixed in one function 
            call.
        degrees: bool, default is False
            Copy of ``scipy.spatial.transform.Rotation.as_euler``. Returned angles 
            are in degrees if this flag is True, else they are in radians.

        Returns
        -------
        (N, 3) ndarray
            Euler angles.
        """
        seq = EulerAxes(seq).value
        seq = _translate_euler(seq)
        return self._rotator.as_euler(seq, degrees=degrees)[..., ::-1]
    
    
    def quaternion(self) -> np.ndarray:
        """
        Calculate quaternions that transforms a source vector to vectors that 
        belong to the object.
        
        Returns
        -------
        (N, 4) ndarray
            Quaternions.
        """
        return self._rotator.as_quat()


    def rotvec(self) -> np.ndarray:
        """
        Calculate rotation vectors that transforms a source vector to vectors that 
        belong to the object.
    
        Returns
        -------
        (N, 3) ndarray
            Rotation vectors.
        """
        return self._rotator.as_rotvec()
    
    
    def translate(self, shifts: ArrayLike, copy: bool = True) -> Molecules:
        """
        Translate molecule positions by ``shifts``. 
        
        Shifts are applied in world coordinates, not internal coordinates of every molecules.
        If molecules should be translated in their own coordinates, such as translating toward 
        y-direction of each molecules by 1.0 nm, use ``translate_internal`` instead. Translation
        operation does not convert molecule orientations.

        Parameters
        ----------
        shifts : (3,) or (N, 3) array
            Spatial shift of molecules.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing instance.

        Returns
        -------
        Molecules
            Instance with updated positional coordinates.
        """
        coords = self._pos + shifts
        if copy:
            out = self.__class__(coords, self._rotator)
        else:
            self._pos = coords
            out = self
        return out
    
    def translate_internal(self, shifts: ArrayLike, copy: bool = True) -> Molecules:
        """
        Translate molecule positions by ``shifts``, in their own coordinates separately. 
        
        Shifts are applied in world coordinates, not internal coordinates of every molecules.
        If molecules should be translated in their own coordinates, such as translating toward 
        y-direction of each molecules by 1.0 nm, use ``translate_internal`` instead. Translation
        operation does not convert molecule orientations.

        Parameters
        ----------
        shifts : (3,) or (N, 3) array
            Spatial shift of molecules.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing instance.

        Returns
        -------
        Molecules
            Instance with updated positional coordinates.
        """        
        world_shifts = self._rotator.apply(shifts)
        return self.translate(world_shifts, copy=copy)
    
    
    def rotate_by_rotvec_internal(self, vector: ArrayLike, copy: bool = True) -> Molecules:
        """
        Rotate molecules using rotation vector, whose components are calculated in the molecule-coordinate.

        Parameters
        ----------
        vector : ArrayLike
            Rotation vector(s).
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing instance.

        Returns
        -------
        Molecules
            Instance with updated angles.
        """        
        vector = np.atleast_2d(vector)
        vec_x = self.x
        vec_y = self.y
        vec_z = -np.cross(vec_x, vec_y, axis=1)
        world_rotvec = (
            vec_z * vector[:, 0][:, np.newaxis]
            + vec_y * vector[:, 1][:, np.newaxis]
            + vec_x * vector[:, 2][:, np.newaxis]
        )
        return self.rotate_by_rotvec(world_rotvec, copy=copy)
    
    
    def rot180(self, axis: str = "z", copy: bool = True) -> Molecules:
        if axis == "x":
            quat = [0., 0., 0., 1.]
        elif axis == "y":
            quat = [0., 0., 1., 0.]
        elif axis == "z":
            quat = [0., 1., 0., 0.]
        else:
            raise ValueError("'axis' must be 'x', 'y' or 'z'.")
        mol = self.rotate_by_quaternion(np.array(quat), copy=copy)
        return mol
    
    def rotate_by_matrix(self, matrix: ArrayLike, copy: bool = True) -> Molecules:
        """
        Rotate molecules using rotation matrices, **with their position unchanged**.

        Parameters
        ----------
        matrix : ArrayLike
            Rotation matrices, whose length must be same as the number of molecules.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing instance.

        Returns
        -------
        Molecules
            Instance with updated orientation.
        """
        rotator = Rotation.from_matrix(matrix)
        return self.rotate_by(rotator, copy)
    
    
    def rotate_by_quaternion(self, quat: ArrayLike, copy: bool = True) -> Molecules:
        """
        Rotate molecules using quaternions, **with their position unchanged**.

        Parameters
        ----------
        quat : ArrayLike
            Rotation quaternion.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing instance.

        Returns
        -------
        Molecules
            Instance with updated orientation.
        """
        rotator = Rotation.from_quat(quat)
        return self.rotate_by(rotator, copy)
    
    
    def rotate_by_euler_angle(self, 
                              angles: ArrayLike,
                              seq: str | EulerAxes = EulerAxes.ZXZ,
                              degrees: bool = False,
                              copy: bool = True) -> Molecules:
        """
        Rotate molecules using Euler angles, **with their position unchanged**.

        Parameters
        ----------
        angles: array-like
            Euler angles of rotation.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing instance.

        Returns
        -------
        Molecules
            Instance with updated orientation.
        """
        rotator = from_euler(angles, seq, degrees)
        return self.rotate_by(rotator, copy)
    
    
    def rotate_by_rotvec(self, vector: ArrayLike, copy: bool = True) -> Molecules:
        """
        Rotate molecules using rotation vectors, **with their position unchanged**.

        Parameters
        ----------
        vector: array-like
            Rotation vectors.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing instance.

        Returns
        -------
        Molecules
            Instance with updated orientation.
        """
        rotator = Rotation.from_rotvec(vector)
        return self.rotate_by(rotator, copy)
    
    
    def rotate_by(self, rotator: Rotation, copy: bool = True) -> Molecules:
        """
        Rotate molecule with a ``Rotation`` object.
        
        Note that ``Rotation`` object satisfies following equation.
        
        .. code-block::python
        
            rot1.apply(rot2.apply(v)) == (rot1*rot2).apply(v)

        Parameters
        ----------
        rotator : Rotation
            Molecules will be rotated by this object.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing instance.

        Returns
        -------
        Molecules
            Instance with updated orientation.
        """
        rot = rotator * self._rotator
        if copy:
            out = self.__class__(self._pos, rot)
        else:
            self._rotator = rot
            out = self
        return out


def _normalize(a: np.ndarray) -> np.ndarray:
    """Normalize vectors to length 1. Input must be (N, 3)."""
    return a / np.sqrt(np.sum(a**2, axis=1))[:, np.newaxis]

def _extract_orthogonal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Extract component of b orthogonal to a."""
    a_norm = _normalize(a)
    return b - np.sum(a_norm * b, axis=1)[:, np.newaxis] * a_norm

def _translate_euler(seq: str) -> str:
    table = str.maketrans({"x": "z", "z": "x", "X": "Z", "Z": "X"})
    return seq[::-1].translate(table)


def from_euler(
    angles: np.ndarray,
    seq: str | EulerAxes = EulerAxes.ZXZ,
    degrees: bool = False
) -> Rotation:
    """Create a rotator from Euler angles using zyx-coordinate system."""
    seq = _translate_euler(EulerAxes(seq).value)
    return Rotation.from_euler(seq, angles[..., ::-1], degrees)

def axes_to_rotator(z, y) -> Rotation:
    ref = _normalize(np.atleast_2d(y))
    
    n = ref.shape[0]
    yx = np.arctan2(ref[:, 2], ref[:, 1])
    zy = np.arctan(-ref[:, 0]/np.abs(ref[:, 1]))
    
    rot_vec_yx = np.zeros((n, 3))
    rot_vec_yx[:, 0] = yx
    rot_yx = Rotation.from_rotvec(rot_vec_yx)
    
    rot_vec_zy = np.zeros((n, 3))
    rot_vec_zy[:, 2] = zy
    rot_zy = Rotation.from_rotvec(rot_vec_zy)
    
    rot1 = rot_yx * rot_zy
    
    if z is None:
        return rot1
    
    vec = _normalize(np.atleast_2d(_extract_orthogonal(ref, z)))
    
    vec_trans = rot1.apply(vec, inverse=True)   # in zx-plane
    
    thetas = np.arctan2(vec_trans[..., 0], vec_trans[..., 2]) - np.pi/2
    
    rot_vec_zx = np.zeros((n, 3))
    rot_vec_zx[:, 1] = thetas
    rot2 = Rotation.from_rotvec(rot_vec_zx)
    
    return rot1 * rot2
