from __future__ import annotations
from functools import wraps
from napari.layers import Points
from acryo import Molecules
import polars as pl

class MoleculesLayer(Points):
    _type_string = "points"

    @wraps(Points.__init__)
    def __init__(self, data, **kwargs):
        if not isinstance(data, Molecules):
            raise TypeError('data must be a Molecules object')
        self._molecules = data
        super().__init__(data.pos, **kwargs)
        if data.features is not None:
            self.features = data.features
    
    @property
    def molecules(self) -> Molecules:
        return self._molecules

    @molecules.setter
    def molecules(self, mole: Molecules):
        if not isinstance(mole, Molecules):
            raise TypeError('Must be a Molecules object')
        self.data = mole.pos
        self._molecules = mole

    @property
    def features(self):
        return Points.features.fget(self)

    @features.setter
    def features(self, features):
        if isinstance(features, pl.DataFrame):
            df = features.to_pandas()
        else:
            df = features
        Points.features.fset(self, df)
        self._molecules.features = df
