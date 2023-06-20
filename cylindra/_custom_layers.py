from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple
import weakref
import polars as pl
import numpy as np
from acryo import Molecules
from napari.layers import Points, Labels
from napari.utils.status_messages import generate_layer_coords_status
from cylindra.const import MoleculesHeader as Mole

if TYPE_CHECKING:
    import impy as ip
    from cylindra.components import BaseComponent, CylSpline
    from napari.utils import Colormap


class ColormapInfo(NamedTuple):
    """Information about a colormap."""

    cmap: Colormap
    clim: tuple[float, float]
    name: str

    def map(self, x):
        l, h = self.clim
        return self.cmap.map((x - l) / (h - l))


class _FeatureBoundLayer:
    def get_status(
        self,
        position: tuple | None = None,
        *,
        view_direction: np.ndarray | None = None,
        dims_displayed: list[int] | None = None,
        world: bool = False,
    ) -> dict:
        if position is not None:
            value = self.get_value(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )
        else:
            value = None

        source_info = self._get_source_info()
        source_info["coordinates"] = generate_layer_coords_status(position, value)

        # if this points layer has properties
        properties = self._get_properties(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        if properties:
            msgs = list[str]()
            nchars = 0
            for prop in properties:
                msgs.append(prop)
                nchars += len(prop)
                if nchars > 80:
                    msgs.append("...")
                    break
            source_info["coordinates"] += "; " + ", ".join(msgs)

        return source_info

    def _get_properties(
        self,
        position,
        *,
        view_direction: np.ndarray | None = None,
        dims_displayed: list[int] | None = None,
        world: bool = False,
    ) -> list:
        if self.features.shape[1] == 0:
            return []

        value = self.get_value(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        # if the cursor is not outside the image or on the background
        if value is None or value > self.data.shape[0]:
            return []

        out = []
        for k, col in self.features.items():
            if k == "index" or len(col) <= value:
                continue
            val = col[value]
            if isinstance(val, np.floating) and not np.isnan(val):
                if abs(val) > 1e4:
                    out.append(f"{k}: {val:.3e}")
                out.append(f"{k}: {val:.3f}")
            else:
                out.append(f"{k}: {val}")

        return out


class MoleculesLayer(_FeatureBoundLayer, Points):
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
        features = data.features
        if features is not None and len(features) > 0:
            self.features = features

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
    def source_component(self, obj: BaseComponent | None):
        from cylindra.components._base import BaseComponent

        if obj is None:
            self._source_component = None
        elif not isinstance(obj, BaseComponent):
            raise TypeError("Must be a CylSpline object")
        else:
            self._source_component = weakref.ref(obj)

    @property
    def source_spline(self) -> CylSpline | None:
        from cylindra.components import CylSpline

        src = self.source_component
        if not isinstance(src, CylSpline):
            return None
        return src

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
        column = self.molecules.features[name]
        clim = tuple(clim)
        cmap = _normalize_colormap(cmap_input)
        if column.dtype in pl.INTEGER_DTYPES:
            cmin, cmax = clim
            arr = (column.cast(pl.Float32).clip(cmin, cmax) - cmin) / (cmax - cmin)
            colors = cmap.map(arr)
            self.face_color = self.edge_color = colors
        elif column.dtype in pl.FLOAT_DTYPES:
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


class CylinderLabels(_FeatureBoundLayer, Labels):
    _type_string = "labels"

    def __init__(self, data, **kwargs):
        self._colormap_info: ColormapInfo | None = None
        super().__init__(data, **kwargs)

    def set_colormap(self, name: str, clim: tuple[float, float], cmap_input: Any):
        color = {
            0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            None: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
        clim = tuple(clim)
        lim0, lim1 = clim
        seq = self.features[name][1:]  # skip background
        cmap = _normalize_colormap(cmap_input)
        for i, value in enumerate(seq):
            color[i + 1] = cmap.map((value - lim0) / (lim1 - lim0))
        self.color = color
        self._colormap_info = ColormapInfo(cmap, clim, name)
        return None

    @property
    def colormap_info(self) -> ColormapInfo | None:
        """Colormap information."""
        return self._colormap_info


def _normalize_colormap(cmap) -> Colormap:
    """Normalize the input to a napari Colormap object."""
    from napari.utils.colormaps import Colormap, ensure_colormap

    if isinstance(cmap, Colormap):
        return cmap
    if isinstance(cmap, str):
        return ensure_colormap(cmap)

    cmap = dict(cmap)
    if 0.0 not in cmap:
        cmap[0.0] = cmap[min(cmap.keys())]
    if 1.0 not in cmap:
        cmap[1.0] = cmap[max(cmap.keys())]
    return Colormap(list(cmap.values()), controls=list(cmap.keys()))
