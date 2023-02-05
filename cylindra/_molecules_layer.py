from __future__ import annotations
from functools import wraps
import numpy as np
import polars as pl
from acryo import Molecules
from napari.layers import Points

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
        """The underlying Molecules object."""
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

    
    def set_colormap(
        self,
        name: str,
        clim: tuple[float, float], 
        cmap_input,
    ):
        """
        Set colormap to a molecules layer.
        
        
        """
        from polars.datatypes import IntegralType, FractionalType
        from napari.utils import Colormap

        column = self.molecules.features[name]
        if isinstance(cmap_input, Colormap):
            cmap = cmap_input
        else:
            cmap = Colormap(cmap_input, name="MoleculeFeatures")
        if issubclass(column.dtype, IntegralType):
            cmin, cmax = clim
            arr = (column.cast(pl.Float32).clip(cmin, cmax) - cmin) / (cmax - cmin)
            colors = cmap.map(arr)
            self.face_color = self.edge_color = colors
        elif issubclass(column.dtype, FractionalType):
            self.face_color = self.edge_color = column.name
            self.face_colormap = self.edge_colormap = cmap
            self.face_contrast_limits = self.edge_contrast_limits = clim
        else:
            raise ValueError(f"Cannot paint by feature {column.name} of type {column.dtype}.")
        self.refresh()