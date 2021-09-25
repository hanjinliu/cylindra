from __future__ import annotations
import numpy as np
from scipy.interpolate import splprep, splev

class Spline3D:
    def __init__(self, coords:np.ndarray, k=3, s=None):
        self.tck, self.u = splprep(coords.T, k=k, s=s)
        
    @classmethod
    def prep(self, t, c, u):
        self.tck = (t, c, 3)
        self.u = u
    
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