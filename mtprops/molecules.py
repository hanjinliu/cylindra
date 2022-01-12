from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy import linalg as npl
from .const import EulerAxes

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation

    
class Molecules:
    """
    Object that represents multiple orientation and position of molecules. Orientation
    is represented by `scipy.spatial.transform.Rotation`. All the vectors are zyx-order.
    """
    def __init__(self, pos: np.ndarray, rot: "Rotation"):
        pos = np.atleast_2d(pos)
        
        if pos.shape[1] != 3:
            raise ValueError("Shape of pos must be (N, 3).")
        elif pos.shape[0] != len(rot):
            raise ValueError("Length mismatch.")
        
        self._pos = pos
        self._rotator = rot
    
    @classmethod
    def from_axes(cls, 
                  pos: np.ndarray, 
                  z: np.ndarray | None = None, 
                  y: np.ndarray | None = None,
                  x: np.ndarray | None = None) -> Molecules:
        pos = np.atleast_2d(pos)
        
        if sum((_ax is not None) for _ax in [z, y, x]) != 2:
            raise TypeError("You must specify two out of z, y, and x.")
        
        if z is None:
            x = np.atleast_2d(x)
            y = np.atleast_2d(y)
            z = np.cross(x, y, axis=0)
        elif y is None:
            z = np.atleast_2d(z)
            x = np.atleast_2d(x)
            y = np.cross(z, x, axis=0)
        
        vec = _normalize(np.atleast_2d(z))
        ref = np.atleast_2d(_normalize(_extract_orthogonal(vec, y)))
        
        if not (pos.shape == vec.shape == ref.shape):
            raise ValueError(f"Mismatch in shapes: {pos.shape}, {vec.shape}, {ref.shape}.")
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"Shape of the input arrays must be (N, 3).")
        
        mat1 = []
        for r in ref:
            mat_ = _vector_to_rotation_matrix(-r)[..., :3, :3]
            mat1.append(mat_)
        mat1 = npl.inv(np.stack(mat1))
        
        vec_trans = np.einsum("ij,ijk->ik", vec, mat1) # in zx-plane
        
        mat2 = []
        thetas = np.arctan2(vec_trans[..., 0], vec_trans[..., 2]) - np.pi/2
        for theta in thetas:
            cos = np.cos(theta)
            sin = np.sin(theta)
            rotation_zx = np.array([[cos, 0.,-sin],
                                    [ 0., 1.,  0.],
                                    [sin, 0., cos]])
            mat2.append(rotation_zx)
            
        mat2 = npl.inv(np.stack(mat2))
        mat = np.einsum("ijk,ikl->ijl", mat1, mat2)
        
        from scipy.spatial.transform import Rotation
        rotator = Rotation.from_matrix(mat)
        return cls(pos, rotator)


    def __len__(self) -> int:
        return self._pos.shape[0]

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
        """Vectors of x-axis."""
        return self._rotator.apply([1., 0., 0.])

        
    def rot_matrix(self) -> np.ndarray:
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
        table = str.maketrans({"x": "z", "z": "x", "X": "Z", "Z": "X"})
        seq = seq.translate(table)
        return self._rotator.as_euler(seq, degrees=degrees)
    
    
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


    def rot_vector(self) -> np.ndarray:
        """
        Calculate rotation vectors that transforms a source vector to vectors that 
        belong to the object.
    
        Returns
        -------
        (N, 3) ndarray
            Rotation vectors.
        """
        return self._rotator.as_rotvec()


def _normalize(a: np.ndarray) -> np.ndarray:
    """Normalize vectors to length 1. Input must be (N, 3)."""
    return a / np.sqrt(np.sum(a**2, axis=1))[:, np.newaxis]

def _extract_orthogonal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Extract component of b orthogonal to a."""
    a_norm = _normalize(a)
    return b - np.sum(a_norm * b, axis=1)[:, np.newaxis] * a_norm


def _vector_to_rotation_matrix(ds: np.ndarray):
    xy = np.arctan2(ds[2], -ds[1])
    zy = np.arctan(-ds[0]/np.abs(ds[1]))
    cos = np.cos(xy)
    sin = np.sin(xy)
    rotation_yx = np.array([[1.,  0.,   0., 0.],
                            [0., cos, -sin, 0.],
                            [0., sin,  cos, 0.],
                            [0.,  0.,   0., 1.]],
                            dtype=np.float32)
    cos = np.cos(zy)
    sin = np.sin(zy)
    rotation_zy = np.array([[cos, -sin, 0., 0.],
                            [sin,  cos, 0., 0.],
                            [ 0.,   0., 1., 0.],
                            [ 0.,   0., 0., 1.]],
                            dtype=np.float32)

    mx = rotation_zy.dot(rotation_yx)
    mx[-1, :] = [0, 0, 0, 1]
    return np.ascontiguousarray(mx)