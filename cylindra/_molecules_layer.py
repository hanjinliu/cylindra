from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple
import weakref
import polars as pl
from acryo import Molecules
from napari.layers import Points
from cylindra.const import MoleculesHeader as Mole

if TYPE_CHECKING:
    import impy as ip
    from cylindra.components._base import BaseComponent
    from napari.utils import Colormap


class ColormapInfo(NamedTuple):
    cmap: Colormap
    clim: tuple[float, float]
    name: str


class MoleculesLayer(Points):
    """
    An extended version of napari Points layers.

    This layer contains a Molecules object as its data source.
    """

    _type_string = "points"

    def __init__(self, data: Molecules, **kwargs):
        if not isinstance(data, Molecules):
            raise TypeError("data must be a Molecules object")
        self._molecules = data
        self._colormap_info: ColormapInfo | None = None
        self._source_component: weakref.ReferenceType[BaseComponent] | None = None
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
            raise TypeError("Must be a Molecules object")
        self.data = mole.pos
        self._molecules = mole
        Points.features.fset(self, mole.features.to_pandas())

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

    @property
    def source_component(self) -> BaseComponent | None:
        """The source tomographic component object."""
        if self._source_component is None:
            return None
        return self._source_component()

    @source_component.setter
    def source_component(self, obj: BaseComponent):
        from cylindra.components._base import BaseComponent

        if not isinstance(obj, BaseComponent):
            raise TypeError("Must be a CylSpline object")
        self._source_component = weakref.ref(obj)

    @property
    def colormap_info(self) -> ColormapInfo | None:
        """Colormap information."""
        return self._colormap_info

    def set_colormap(
        self,
        name: str,
        clim: tuple[float, float],
        cmap_input: Any,
    ):
        """
        Set colormap to a molecules layer.

        Parameters
        ----------
        name : str
            Feature name from which colormap will be generated.
        clim : (float, float)
            Colormap contrast limits.
        cmap_input : Any
            Any object that can be converted to a Colormap object.
        """
        from napari.utils import Colormap

        column = self.molecules.features[name]
        if isinstance(cmap_input, Colormap):
            cmap = cmap_input
        else:
            cmap = Colormap(cmap_input, name="MoleculeFeatures")
        if column.dtype.__name__[0] in "IU":
            cmin, cmax = clim
            arr = (column.cast(pl.Float32).clip(cmin, cmax) - cmin) / (cmax - cmin)
            colors = cmap.map(arr)
            self.face_color = self.edge_color = colors
        elif column.dtype.__name__[0] == "F":
            self.face_color = self.edge_color = column.name
            self.face_colormap = self.edge_colormap = cmap
            self.face_contrast_limits = self.edge_contrast_limits = clim
        else:
            raise ValueError(
                f"Cannot paint by feature {column.name} of type {column.dtype}."
            )
        self._colormap_info = ColormapInfo(cmap, clim, name)
        self.refresh()
        return None

    def to_coordinates(self, npf: int | None = None) -> ip.ImgArray:
        """Convert point coordinates of a Points layer into a structured array."""
        import impy as ip

        if npf is None:
            npf = self.molecules.features[Mole.pf].max() + 1
        data = self.data.reshape(-1, npf, 3)

        data = ip.asarray(data, name=self.name, axes=["L", "PF", "dim"])
        data.axes["dim"].labels = ("z", "y", "x")
        return data
