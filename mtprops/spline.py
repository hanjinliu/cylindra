from __future__ import annotations
import numpy as np
from scipy.interpolate import splprep, splev, interp1d

class Spline3D:
    tck: tuple[np.ndarray, list[np.ndarray], int]
    u: np.ndarray
    def __init__(self, coords:np.ndarray=None, k=3, s=None):
        if coords is None:
            return None
        
        npoints = coords.shape[0]
        if npoints < 4:
            lin = interp1d(np.linspace(0, 1, npoints), coords.T)
            coords = lin(np.linspace(0,1,4)).T
        self.tck, self.u = splprep(coords.T, k=k, s=s)
        
    @classmethod
    def prep(cls, t, c, u):
        self = cls()
        self.tck = (t, c, 3)
        self.u = u
        return self
    
    def __call__(self, u:np.ndarray, der:int=0):
        coords = splev(u, self.tck, der=der)
        return np.stack(coords, axis=1)
    
    def partition(self, n:int, der:int=0):
        u = np.linspace(0, 1, n)
        return self(u, der)
    
    def length(self, start:float=0, stop:float=1, nknots:int=100):
        """
        Approximate the length of B-spline between [start, stop] by partitioning
        the spline with 'nknots' knots.
        """
        u = np.linspace(start, stop, nknots)
        dz, dy, dx = map(np.diff, splev(u, self.tck, der=0))
        return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))