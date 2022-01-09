from __future__ import annotations
from typing import TYPE_CHECKING, Iterator
import numpy as np
from .const import EulerAxes

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation
    
class VectorField3D:
    """
    3-D vector field class equipped with methods for vector rotations.
    Each object retains its position in world coordinate system and its vector.
    """
    def __init__(self, pos: np.ndarray, vec: np.ndarray):
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"Shape of the position array must be (N, 3), got {pos.shape}")
        if vec.ndim != 2 or vec.shape[1] != 3:
            raise ValueError(f"Shape of the vector array must be (N, 3), got {vec.shape}")
        if pos.shape != vec.shape:
            raise ValueError("Shape mismatch in positions and vectors.")
        
        self._pos = pos
        self._vec = vec
    
    
    def __array__(self, dtype=None) -> np.ndarray:
        return self.vectors

    
    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._vec)
    
    
    def __len__(self) -> int:
        return self._vec.shape[0]
    
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} object with {len(self)} vectors"
    
    
    def __neg__(self) -> VectorField3D:
        """Invert vectors."""
        return self.invert()
    
    
    def __getitem__(self, key):
        return self._vec[key]
    
    @property
    def starts(self) -> np.ndarray:
        """Start points."""
        return self._pos
    
    @property
    def ends(self) -> np.ndarray:
        """End points."""
        return self._pos + self._vec
    
    @property
    def vectors(self) -> np.ndarray:
        """Vectors."""
        return self._vec
    
    
    def linear_transform(self, mtx: np.ndarray) -> VectorField3D:
        """Linear-transform every vector while kept start points unchanged."""
        mtx = np.asarray(mtx)
        if mtx.shape != (3, 3):
            raise ValueError(f"Transformation matrix must be (3, 3), got {mtx.shape}.")
        vec = self._vec @ mtx
        return VectorField3D(self._pos, vec)
    
    
    def affine_transform(self, mtx: np.ndarray) -> VectorField3D:
        """Affine-transform every vector while kept start points unchanged."""
        mtx = np.asarray(mtx)
        if mtx.shape != (4, 4):
            raise ValueError(f"Transformation matrix must be (4, 4), got {mtx.shape}.")
        input = np.concatenate([self._vec, np.ones(self._vec.shape[0])], axis=1)
        vec = input @ mtx
        return VectorField3D(self._pos, vec[:, :3])
    
    
    def invert(self) -> VectorField3D:
        """Invert vectors."""
        return self.__class__(self._pos, -self._vec)
    
    
    def normalize(self) -> VectorField3D:
        """Normalize vectors."""
        norm = np.sqrt(np.sum(self._vec**2, axis=1))
        return self.__class__(self._pos, self._vec/norm)
    
    
    def rot_matrix(self, src_vector: np.ndarray | str = "z") -> np.ndarray:
        """
        Calculate rotation matrices that transforms a source vector to vectors that 
        belong to the object.

        Parameters
        ----------
        src_vector : (3,) ndarray or str of {"z", "y", "x"}
            Source vector. Three components must be defined in the world coordinate.
            If a string is given, a unit vector pointing to that axis will be used.

        Returns
        -------
        (N, 3, 3) ndarray
            Rotation matrices. Should satisfy ``src_vector @ output[i] == self[i]``.
        """
        if isinstance(src_vector, str):
            axis = src_vector.lower()
            if axis == "z":
                src_vector = np.array([1., 0., 0.], dtype=np.float32)
            elif axis == "y":
                src_vector = np.array([0., 1., 0.], dtype=np.float32)
            elif axis == "x":
                src_vector = np.array([0., 0., 1.], dtype=np.float32)
            else:
                raise ValueError("'src_vector' must be 'z', 'y', 'x' or an array.")
        
        else:
            src_vector = np.asarray(src_vector)
            if src_vector.shape != (3,):
                raise ValueError(
                    f"Shape of 'src_vector' must be (3,) but got {src_vector.shape}."
                    )
        
        return _vec_to_vec_rotation_matrix(src_vector, np.array(self))
    
    
    def euler_angle(self, 
                    src_vector: np.ndarray | str = "z",
                    seq: str | EulerAxes = EulerAxes.ZXZ,
                    degrees: bool = False) -> np.ndarray:
        """
        Calculate Euler angles that transforms a source vector to vectors that 
        belong to the object.

        Parameters
        ----------
        src_vector : (3,) ndarray or str of {"z", "y", "x"}
            Source vector. Three components must be defined in the world coordinate.
            If a string is given, a unit vector pointing to that axis will be used.
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
        rot = self._get_rotator(src_vector)
        return rot.as_euler(seq, degrees=degrees)
    
    
    def quaternion(self, src_vector: np.ndarray | str = "z") -> np.ndarray:
        """
        Calculate quaternions that transforms a source vector to vectors that 
        belong to the object.

        Parameters
        ----------
        src_vector : (3,) ndarray or str of {"z", "y", "x"}
            Source vector. Three components must be defined in the world coordinate.
            If a string is given, a unit vector pointing to that axis will be used.

        Returns
        -------
        (N, 4) ndarray
            Quaternions.
        """
        rot = self._get_rotator(src_vector)
        return rot.as_quat()


    def rot_vector(self, src_vector: np.ndarray | str = "z") -> np.ndarray:
        """
        Calculate rotation vectors that transforms a source vector to vectors that 
        belong to the object.

        Parameters
        ----------
        src_vector : (3,) ndarray or str of {"z", "y", "x"}
            Source vector. Three components must be defined in the world coordinate.
            If a string is given, a unit vector pointing to that axis will be used.

        Returns
        -------
        (N, 3) ndarray
            Rotation vectors.
        """
        rot = self._get_rotator(src_vector)
        return rot.as_rotvec()
    
    
    def _get_rotator(self, src_vector: np.ndarray | str) -> "Rotation":
        mtxs = self.rot_matrix(src_vector)
        
        from scipy.spatial.transform import Rotation
        
        return Rotation.from_matrix(mtxs)



def _vec_to_vec_rotation_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Calculate the 3-D rotation matrix that transforms vector ``src`` to ``dst``.
    
    References
    ----------
    - https://ja.stackoverflow.com/questions/55865/2%E3%81%A4%E3%81%AE3%E6%AC%A1%E5%85%83%E5%AE%9F%E6%95%B0%E5%80%A4%E3%83%99%E3%82%AF%E3%83%88%E3%83%ABx%E3%81%A8y%E3%81%8C%E4%B8%8E%E3%81%88%E3%82%89%E3%82%8C%E3%81%9F%E6%99%82-x%E3%81%8B%E3%82%89y%E3%81%B8%E5%9B%9E%E8%BB%A2%E3%81%99%E3%82%8B%E8%A1%8C%E5%88%97%E3%82%92%E6%B1%82%E3%82%81%E3%81%9F%E3%81%84
    """
    I = np.eye(3)
    n = dst.shape[0]
    out = np.empty((n, 3, 3), dtype=np.float32)
    for i in range(n):
        d = dst[i]
        if np.all(src == d):
            out[i, :, :] = I
        elif np.all(src == -d):
            out[i, :, :] = -I
        else:
            s = src + d
            out[i, :, :] = 2.0 * np.outer(s, s) / np.dot(s, s) - I
    return out
