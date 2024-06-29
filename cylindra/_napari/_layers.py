from __future__ import annotations

import warnings
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, TypedDict

import numpy as np
import polars as pl
from acryo import Molecules
from napari.layers import Points, Surface
from napari.utils import Colormap
from napari.utils.events import Event
from napari.utils.status_messages import generate_layer_coords_status

from cylindra._config import get_config
from cylindra.const import POLARS_FLOAT_DTYPES, POLARS_INTEGER_DTYPES
from cylindra.const import MoleculesHeader as Mole
from cylindra.utils import assert_column_exists, str_color

if TYPE_CHECKING:
    from cylindra.components import BaseComponent, CylSpline
    from cylindra.components.landscape import Landscape


class ColormapInfo(NamedTuple):
    """Information about a colormap."""

    cmap: Colormap
    clim: tuple[float, float]
    name: str

    def map(self, x: np.ndarray) -> np.ndarray:
        l, h = self.clim
        return self.cmap.map((x - l) / (h - l))

    def to_list(self) -> list[tuple[float, str]]:
        out = []
        for cont, cols in zip(self.cmap.controls, self.cmap.colors, strict=False):
            out.append((cont, str_color(cols)))
        return out


class CmapDict(TypedDict):
    by: str
    limits: tuple[float, float]
    cmap: Any


class _FeatureBoundLayer:
    _property_filter: Callable[[str], bool]

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
            if k == "index" or len(col) <= value or not self._property_filter(k):
                continue
            val = col[value]
            if isinstance(val, np.floating) and not np.isnan(val):
                if abs(val) > 1e4:
                    out.append(f"{k}: {val:.3e}")
                out.append(f"{k}: {val:.3f}")
            else:
                out.append(f"{k}: {val}")

        return out


class _SourceBoundLayer:
    _source_component: weakref.ReferenceType[BaseComponent] | None = None

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
        """The source component but limited to splines."""
        from cylindra.components import CylSpline

        src = self.source_component
        if not isinstance(src, CylSpline):
            return None
        return src


class MoleculesLayer(_FeatureBoundLayer, Points, _SourceBoundLayer):
    """
    An extended version of napari Points layers.

    This layer contains a Molecules object as its data source.
    """

    _type_string = "points"

    def __init__(self, data: Molecules, **kwargs):
        if not isinstance(data, Molecules):
            raise TypeError("data must be a Molecules object")
        self._molecules = data
        self._colormap_info: ColormapInfo | str = "lime"
        self._point_size = float(kwargs.get("size", get_config().point_size))
        if isinstance(col := kwargs.get("face_color"), str):
            self._colormap_info = col
        self._source_component: weakref.ReferenceType[BaseComponent] | None = None
        self._property_filter = lambda _: True
        self._old_name: str | None = None  # for undo/redo
        self._undo_renaming = False
        self._view_ndim = 3
        super().__init__(data.pos, **kwargs)
        self.editable = False
        self.events.add(point_size=Event, view_ndim=Event)
        features = data.features
        if features is not None and len(features) > 0:
            self.features = features

    @classmethod
    def construct(
        cls,
        mol: Molecules,
        name: str,
        source: BaseComponent | None = None,
        metadata: dict[str, Any] = {},
        cmap: CmapDict | None = None,
        **kwargs,
    ):
        app_cfg = get_config()
        kw = {
            "size": app_cfg.point_size,
            "face_color": app_cfg.molecules_color,
            "out_of_slice_display": True,
        }
        kw.update(**kwargs)
        layer = MoleculesLayer(mol, name=name, metadata=metadata.copy(), **kw)
        if source is not None:
            layer.source_component = source
        if cmap is not None:
            layer.set_colormap(**cmap)
        layer.view_ndim = app_cfg.molecules_ndim
        return layer

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
    def name(self) -> str:
        return super().name

    @name.setter
    def name(self, name: str) -> None:
        if self.name == name:
            return None
        self._old_name = self.name
        if not name:
            name = self._basename()
        self._name = str(name)
        self.events.name()

    @property
    def point_size(self) -> float:
        return self._point_size

    @point_size.setter
    def point_size(self, size: float):
        self.size = size
        self._point_size = size
        self.events.point_size(value=size)
        self.refresh()

    @contextmanager
    def _undo_context(self):
        was_renaming = self._undo_renaming
        self._undo_renaming = True
        try:
            yield
        finally:
            self._undo_renaming = was_renaming

    def _rename(self, name: str):
        with self._undo_context():
            self.name = name

    @property
    def colormap_info(self) -> ColormapInfo | str:
        """Colormap information."""
        return self._colormap_info

    def set_colormap(
        self,
        by: str,
        limits: tuple[float, float] | None = None,
        cmap: Any | None = None,
    ):
        """
        Set colormap to a molecules layer.

        Parameters
        ----------
        by : str
            Feature name from which colormap will be generated.
        limits : (float, float)
            Colormap contrast limits. Use min/max by default.
        cmap : Any
            Any object that can be converted to a Colormap object.
        """
        assert_column_exists(self.molecules.features, by)
        column = self.molecules.features[by]
        if limits is None:
            seq = column.filter(column.is_finite())
            limits = (seq.min(), seq.max())
        else:
            limits = tuple(limits)
        if cmap is None:
            cmap = {0: "black", 1: "white"}
        _cmap = _normalize_colormap(cmap)
        if column.dtype in POLARS_INTEGER_DTYPES:
            cmin, cmax = limits
            arr = (column.cast(pl.Float32).clip(cmin, cmax) - cmin) / (cmax - cmin)
            colors = _cmap.map(arr)
            self.face_color = colors
        elif column.dtype in POLARS_FLOAT_DTYPES:
            self.face_color = column.name
            self.face_colormap = _cmap
            self.face_contrast_limits = limits
            if self._view_ndim == 3:
                self.edge_colormap = _cmap
                self.edge_contrast_limits = limits
        elif column.dtype == pl.Boolean:  # NOTE: since polars>=0.20, `is` fails
            cfalse, ctrue = _cmap.map([0, 1])
            column2d = np.repeat(column.to_numpy()[:, np.newaxis], 4, axis=1)
            col = np.where(column2d, ctrue, cfalse)
            self.face_color = col
        else:
            raise ValueError(
                f"Cannot paint by feature {column.name} of type {column.dtype}."
            )
        self._colormap_info = ColormapInfo(_cmap, limits, by)
        self.refresh()
        return None

    def feature_setter(
        self, features: pl.DataFrame, cmap_info: ColormapInfo | str | None = None
    ):
        def _wrapper():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.features = features
            match cmap_info:
                case str(color):
                    self.face_color = color
                case None:
                    self.face_color = "#000000"
                case info:
                    self.set_colormap(info.name, info.clim, info.cmap)

        return _wrapper

    def face_color_setter(self, color):
        self.face_color = color
        self._colormap_info = str_color(color)
        self.refresh()

    @property
    def view_ndim(self):
        """Get the view dimension."""
        return self._view_ndim

    @view_ndim.setter
    def view_ndim(self, ndim: int):
        """Set the view dimension."""
        match ndim:
            case 2:
                self.shading = "none"
                self.edge_color = "#222222"
            case 3:
                self.shading = "spherical"
                self.edge_color = self.face_color
            case _:
                raise ValueError("ndim must be 2 or 3")
        self.events.view_ndim(value=ndim)
        self._view_ndim = ndim

    @Points.face_color.setter
    def face_color(self, color: Any):
        Points.face_color.fset(self, color)
        if self._view_ndim == 3:
            self.edge_color = color
        if isinstance(color, str):
            self._colormap_info = color
        elif isinstance(color[0], (int, float, np.number)):
            self._colormap_info = str_color(color)

    def regular_shape(self) -> tuple[int, int]:
        """Get the regular shape (long, lat) of the layer."""
        mole = self.molecules
        df = mole.features
        nlon = df[Mole.nth].n_unique()
        nlat = df[Mole.pf].n_unique()
        if nlon * nlat != mole.count():
            raise ValueError("Molecules are not regularly arranged")
        return nlon, nlat


class LandscapeSurface(Surface, _SourceBoundLayer):
    """Surface layer for an energy landscape."""

    _type_string = "surface"

    def __init__(self, landscape: Landscape, level=None, **kwargs):
        kwargs.setdefault("blending", "translucent_no_depth")
        kwargs.setdefault(
            "wireframe", {"visible": True, "color": "crimson", "width": 0.7}
        )
        kwargs.setdefault(
            "colormap",
            Colormap([[0, 0, 0, 0], [0, 0, 0, 0]], controls=[0, 1], name="invisible"),
        )
        self._level_min = landscape.energies.min()
        self._level_max = landscape.energies.max()
        if level is None:
            level = (self._level_max + self._level_min) / 2
        self._resolution = 0.25
        self._energy_level = level
        self._show_min = True
        data = landscape.create_surface(level=level, resolution=0.25)
        super().__init__(data, **kwargs)
        self.events.add(level=Event, resolution=Event, show_min=Event)
        self._landscape = landscape
        self._source_component: weakref.ReferenceType[BaseComponent] | None = None
        self.events.level(value=level)

    @property
    def landscape(self):
        """The landscape object."""
        return self._landscape

    @property
    def molecules(self):
        """Molecules that represent the center/rotation of the landscape."""
        return self._landscape.molecules

    @property
    def level(self):
        """The threshold level of the energy landscape."""
        return self._energy_level

    @level.setter
    def level(self, level: float):
        if level < self._level_min or level > self._level_max:
            raise ValueError(
                f"level must be in range of ({self._level_min}, {self._level_max})"
            )
        self._energy_level = level
        self._update_surface()
        self.events.level(value=level)

    @property
    def show_min(self) -> bool:
        """Whether to show the minimum energy surface."""
        return self._show_min

    @show_min.setter
    def show_min(self, show: bool):
        show = bool(show)
        self._show_min = show
        if show:
            self.wireframe.color = "crimson"
        else:
            self.wireframe.color = "darkblue"
        self._update_surface()
        self.events.show_min(value=show)

    @property
    def resolution(self) -> float:
        """The resolution of the surface."""
        return self._resolution

    @resolution.setter
    def resolution(self, res: float):
        if res <= 0:
            raise ValueError("resolution must be positive")
        self._resolution = res
        self._update_surface()
        self.events.resolution(value=res)

    def _update_surface(self):
        self.data = self._landscape.create_surface(
            level=self._energy_level,
            resolution=self._resolution,
            show_min=self._show_min,
        )
        self.refresh()


def _normalize_colormap(cmap) -> Colormap:
    """Normalize the input to a napari Colormap object."""
    from napari.utils.colormaps import Colormap, ensure_colormap

    if isinstance(cmap, Colormap):
        return cmap
    if isinstance(cmap, str):
        return ensure_colormap(cmap)
    if isinstance(cmap, list) and isinstance(cmap[0], str):
        return Colormap(cmap)
    cmap = dict(cmap)
    if 0.0 not in cmap:
        cmap[0.0] = cmap[min(cmap.keys())]
    if 1.0 not in cmap:
        cmap[1.0] = cmap[max(cmap.keys())]
    colors = list[Any]()
    controls = list[float]()
    for cont, col in sorted(cmap.items(), key=lambda x: x[0]):
        controls.append(cont)
        colors.append(col)
    return Colormap(list(cmap.values()), controls=list(cmap.keys()))
