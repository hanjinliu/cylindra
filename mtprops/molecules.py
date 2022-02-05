from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
import numpy as np
from numpy.typing import ArrayLike
from .const import EulerAxes

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation

    
class Molecules:
    """
    Object that represents multiple orientation and position of molecules. Orientation
    is represented by `scipy.spatial.transform.Rotation`. **All the vectors are zyx-order**.
    """
    def __init__(self, pos: np.ndarray, rot: "Rotation"):
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
    def from_axes(cls, 
                  pos: np.ndarray, 
                  z: np.ndarray | None = None, 
                  y: np.ndarray | None = None,
                  x: np.ndarray | None = None) -> Molecules:
        """Construct molecule cloud with orientation from two of their local axes."""
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
        
        mat1 = _vector_to_rotation_matrix(-ref)
        
        vec_trans = np.einsum("ij,ijk->ik", vec, mat1) # in zx-plane
        
        mat2 = []
        thetas = np.arctan2(vec_trans[..., 0], vec_trans[..., 2]) - np.pi/2
        for theta in thetas:
            cos = np.cos(theta)
            sin = np.sin(theta)
            rotation_zx = np.array([[ cos, 0., sin],
                                    [  0., 1.,  0.],
                                    [-sin, 0., cos]])
            mat2.append(rotation_zx)
            
        mat2 = np.stack(mat2)
        mat = np.einsum("ijk,ikl->ijl", mat1, mat2)
        
        from scipy.spatial.transform import Rotation
        rotator = Rotation.from_matrix(mat)
        return cls(pos, rotator)

    @classmethod
    def from_euler(cls, pos: np.ndarray, angles: np.ndarray, 
                   seq: str | EulerAxes = EulerAxes.ZXZ, degrees: bool = False):
        from scipy.spatial.transform import Rotation
        seq = _translate_euler(EulerAxes(seq).value)
        rotator = Rotation.from_euler(seq, angles[..., ::-1], degrees)
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
        """Vectors of z-axis."""
        return self._rotator.apply([1., 0., 0.])

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
        
        from scipy.spatial.transform import Rotation
        return cls(all_pos, Rotation(all_quat))
        
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
    
    
    def translate(self, shifts: ArrayLike, copy: bool = True) -> Molecules:
        """
        Translate molecule positions by ``shifts``. This operation does not convert
        molecule orientations.

        Parameters
        ----------
        shifts : ArrayLike
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
        from scipy.spatial.transform import Rotation
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
        from scipy.spatial.transform import Rotation
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
        from scipy.spatial.transform import Rotation
        seq = _translate_euler(EulerAxes(seq).value)
        rotator = Rotation.from_euler(seq, angles[..., ::-1], degrees)
        return self.rotate_by(rotator, copy)
    
    
    def rotate_by_rot_vector(self, vector: ArrayLike, copy: bool = True) -> Molecules:
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
        from scipy.spatial.transform import Rotation
        rotator = Rotation.from_rotvec(vector)
        return self.rotate_by(rotator, copy)
    
    
    def rotate_by(self, rotator: "Rotation", copy: bool = True) -> Molecules:
        rot = self._rotator * rotator
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

def _vector_to_rotation_matrix(ds: np.ndarray):
    n = ds.shape[0]
    xy = np.arctan2(ds[:, 2], -ds[:, 1])
    zy = np.arctan(-ds[:, 0]/np.abs(ds[:, 1]))
    
    # In YX plane, rotation matrix should be
    # [[1.,  0.,  0.],
    #  [0., cos, sin],
    #  [0.,-sin, cos]]
    cos = np.cos(xy)
    sin = np.sin(xy)
    rotation_yx = np.zeros((n, 3, 3), dtype=np.float32)
    rotation_yx[:, 0, 0] = 1.
    rotation_yx[:, 1, 1] = rotation_yx[:, 2, 2] = cos
    rotation_yx[:, 2, 1] = -sin
    rotation_yx[:, 1, 2] = sin
    
    # In ZY plane, rotation matrix should be
    # [[1.,  0.,  0.],
    #  [0., cos, sin],
    #  [0.,-sin, cos]]
    cos = np.cos(zy)
    sin = np.sin(zy)
    rotation_zy = np.zeros((n, 3, 3), dtype=np.float32)
    rotation_zy[:, 2, 2] = 1.
    rotation_zy[:, 0, 0] = rotation_zy[:, 1, 1] = cos
    rotation_zy[:, 1, 0] = -sin
    rotation_zy[:, 0, 1] = sin

    return np.einsum("ijk,ikl->ijl", rotation_yx, rotation_zy)
