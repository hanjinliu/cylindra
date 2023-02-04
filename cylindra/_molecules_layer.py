from __future__ import annotations
from functools import wraps
from napari.layers import Points
from acryo import Molecules

class MoleculesLayer(Points):
    _type_string = "points"

    @wraps(Points.__init__)
    def __init__(self, data, **kwargs):
        if not isinstance(data, Molecules):
            raise TypeError('data must be a Molecules object')
        self._molecules = data
        super().__init__(data.pos, **kwargs)
    
    @property
    def molecules(self) -> Molecules:
        return self._molecules

    @molecules.setter
    def molecules(self, mole: Molecules):
        self.data = mole.pos
        self._molecules = mole

    @property
    def features(self):
        return self._molecules.features
    
    @features.setter
    def features(self, features):
        self._molecules.features = features
