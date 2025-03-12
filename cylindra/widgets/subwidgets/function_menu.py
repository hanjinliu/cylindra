import operator
from contextlib import suppress
from typing import Annotated, Any

import impy as ip
import napari
import numpy as np
from magicclass import (
    MagicTemplate,
    impl_preview,
    logging,
    magicclass,
    magicmenu,
    set_design,
    vfield,
)
from magicclass.types import Path
from magicclass.widgets import FloatRangeSlider
from magicgui.types import Separator
from napari.layers import Image, Labels, Layer
from napari.types import LayerDataTuple

from cylindra.components.imscale import ScaleOptimizer
from cylindra.components.seam_search import SeamSearchResult
from cylindra.const import PREVIEW_LAYER_NAME, SEAM_SEARCH_RESULT, FileFilter
from cylindra.utils import set_gpu
from cylindra.widget_utils import FscResult, capitalize
from cylindra.widgets._annotated import assert_layer


def _convert_array(arr: np.ndarray, scale: float) -> ip.ImgArray:
    if not isinstance(arr, ip.ImgArray):
        if len(arr.shape) == 3:
            arr = ip.asarray(arr, axes="zyx")
        else:
            arr = ip.asarray(arr)
        arr.set_scale(xyz=scale, unit="nm")
    return arr


_Logger = logging.getLogger("cylindra")
OPERATORS = [
    ("+", "add"),
    ("-", "sub"),
    ("*", "mul"),
    ("/", "truediv"),
    ("==", "eq"),
    (">=", "ge"),
    (">", "gt"),
    ("<=", "le"),
    ("<", "lt"),
]


@magicmenu(record=False)
class Volume(MagicTemplate):
    """A custom menu that provides useful functions for volumeric data visualization."""

    def __init__(self, viewer: napari.Viewer):
        self._viewer = viewer

    def __post_init__(self):
        self.native.setParent(self._viewer.window.main_menu, self.native.windowFlags())
        self._viewer.window.main_menu.addMenu(self.native)
        self._viewer.layers.events.connect(self.reset_choices)

    @set_design(text=capitalize)
    def binning(
        self, layer: Image, bin_size: Annotated[int, {"min": 1, "max": 16}] = 2
    ) -> LayerDataTuple:
        img = _convert_array(layer.data, layer.scale[-1])
        out = img.binning(binsize=bin_size, check_edges=False)
        translate = list[float]()
        for k, v in img.scale.items():
            if k in ["z", "y", "x"]:
                translate.append((bin_size - 1) / 2 * v)
            else:
                translate.append(0.0)

        return (
            out,
            {
                "scale": out.scale,
                "translate": translate,
                "name": layer.name + "-binning",
                "rendering": layer.rendering,
            },
            "image",
        )

    @set_design(text=capitalize)
    def gaussian_filter(
        self,
        layer: Image,
        sigma: Annotated[float, {"widget_type": "FloatSlider", "max": 5.0, "step": 0.1}] = 1.0,
    ) -> LayerDataTuple:  # fmt: skip
        """Apply Gaussian filter to an image."""
        return self._apply_method(layer, "gaussian_filter", sigma=sigma)

    @impl_preview(gaussian_filter, auto_call=True)
    def _preview_gaussian_filter(self, layer: Image, sigma: float):
        data, kwargs, _ = self._apply_method(layer, "gaussian_filter", sigma=sigma)
        yield from self._preview_context(layer, data, kwargs)

    @set_design(text=capitalize)
    def threshold(
        self,
        layer: Image,
        quantile: Annotated[float, {"widget_type": "FloatSlider", "max": 1.0, "step": 0.01}] = 0.5,
    ) -> LayerDataTuple:  # fmt: skip
        """Apply threshold to an image."""
        thr = np.quantile(layer.data, quantile)
        return self._apply_method(layer, "threshold", thr)

    @impl_preview(threshold, auto_call=True)
    def _preview_threshold(self, layer: Image, quantile: float):
        thr = np.quantile(layer.data, quantile)
        data, kwargs, _ = self._apply_method(layer, "threshold", thr)
        yield from self._preview_context(layer, data, kwargs)

    @set_design(text=capitalize)
    def binary_operation(
        self, layer_1: Image, op: Annotated[str, {"choices": OPERATORS}], layer_2: Image
    ) -> LayerDataTuple:
        img1 = _convert_array(layer_1.data, layer_1.scale[-1])
        img2 = _convert_array(layer_2.data, layer_2.scale[-1])
        out = getattr(operator, op)(img1, img2)
        return (
            out,
            {
                "scale": layer_1.scale,
                "translate": layer_1.translate,
                "name": f"{layer_1.name}-binary_op",
                "rendering": layer_1.rendering,
            },
            "image",
        )

    def _get_seam_searched_layers(self, *_) -> list[Image]:
        if self.parent_viewer is None:
            return []
        return [
            (layer.name, layer)
            for layer in self.parent_viewer.layers
            if SEAM_SEARCH_RESULT in layer.metadata
        ]

    @set_design(text=capitalize)
    def save_seam_search_result(
        self,
        layer: Annotated[Image | str, {"choices": _get_seam_searched_layers}],
        path: Path.Save[FileFilter.CSV],
    ):
        """
        Save seam search result.

        Parameters
        ----------
        layer : str or Image
            Layer that contains seam search result.
        path : Path
            Path to save the result.
        """
        layer = assert_layer(layer, self.parent_viewer)
        result = layer.metadata.get(SEAM_SEARCH_RESULT, None)
        if not isinstance(result, SeamSearchResult):
            raise TypeError("The layer does not have seam search result.")
        return result.to_dataframe().write_csv(path)

    def _get_fsc_layers(self, *_) -> list[Image]:
        if self.parent_viewer is None:
            return []
        return [
            (layer.name, layer)
            for layer in self.parent_viewer.layers
            if "fsc" in layer.metadata
        ]

    @set_design(text=capitalize)
    def save_fsc_result(
        self,
        layer: Annotated[Image, {"choices": _get_fsc_layers}],
        path: Path.Save[FileFilter.DIRECTORY],
        multiple_halfmaps: bool = False,
    ):
        """
        Save FSC result.

        Parameters
        ----------
        layer : str or Image
            Layer that contains seam search result.
        path : Path
            Directory to save the result.
        multiple_halfmaps : bool
            If True, save multiple set of halfmaps if available.
        """
        save_dir = Path(path)
        if not save_dir.exists():
            save_dir.mkdir()
        if not save_dir.is_dir():
            raise ValueError(f"`path` must be a directory, got {save_dir.as_posix()}.")
        _fsc = layer.metadata.get("fsc", None)
        _fsc_halfmaps = layer.metadata.get("fsc_halfmaps", None)
        _fsc_mask = layer.metadata.get("fsc_mask", None)
        if isinstance(_fsc, FscResult):
            _fsc.to_dataframe().write_csv(save_dir / "fsc.csv")
        if _fsc_halfmaps is not None:
            if isinstance(_fsc_halfmaps, tuple) and len(_fsc_halfmaps) == 2:
                img0, img1 = _fsc_halfmaps
                if isinstance(img0, ip.ImgArray) and isinstance(img1, ip.ImgArray):
                    if multiple_halfmaps:
                        for i in range(img0.shape[0]):
                            img0[i].imsave(save_dir / f"halfmap-0_set{i}.mrc")
                            img1[i].imsave(save_dir / f"halfmap-1_set{i}.mrc")
                    else:
                        img0[0].imsave(save_dir / "halfmap-0.mrc")
                        img1[0].imsave(save_dir / "halfmap-1.mrc")
        if isinstance(_fsc_mask, ip.ImgArray):
            _fsc_mask.imsave(save_dir / "mask.mrc")
        return None

    sep0 = Separator

    @set_design(text=capitalize)
    def save_volume(self, layer: Image, path: Path.Save):
        """Save a volume as tif or mrc file."""
        img = layer.data
        if not isinstance(img, ip.ImgArray):
            raise TypeError(f"Use napari built-in menu to save {type(img)}.")
        img.imsave(path)

    @set_design(text=capitalize)
    def save_label_as_mask(self, layer: Labels, path: Path.Save):
        """Save a label as mask."""
        lbl: np.ndarray = layer.data
        if lbl.min() != 0 or np.unique(lbl).size != 2:
            raise ValueError("The label must be binary.")
        if lbl.ndim == 3:
            axes = "zyx"
        else:
            axes = None
        lbl = ip.asarray(lbl, axes=axes, dtype=np.bool_).set_scale(
            xyz=layer.scale[-1], unit="nm"
        )
        lbl.imsave(path)

    sep1 = Separator

    @set_design(text=capitalize)
    def plane_clip(self):
        """Open a plane clipper as an dock widget."""
        widget = PlaneClip()
        self.parent_viewer.window.add_dock_widget(widget, area="right")
        widget._connect_layer()
        return None

    @set_design(text=capitalize)
    def calculate_scale_to_fit(
        self,
        layer: Image,
        target_layer: Image,
        zoom_range: tuple[float, float] = (0.9, 1.1),
        precision: Annotated[float, {"step": 0.0001}] = 0.001,
    ):
        """Calculate the scale of the input layer to fit to the target layer.

        Parameters
        ----------
        layer : Image
            Layer to rescale.
        target_layer : Image
            Target layer to fit.
        zoom_range : (float, float)
            Range of zooming factor.
        precision : float
            Precision of the scale (nm/pixel).
        """
        import matplotlib.pyplot as plt

        img = _convert_array(layer.data, layer.scale[-1])
        target_img = _convert_array(target_layer.data, target_layer.scale[-1])
        opt = ScaleOptimizer(*zoom_range, precision=precision)
        result = opt.fit(img, target_img)
        with _Logger.set_plt():
            plt.figure(figsize=(3.6, 3.6))
            plt.plot(result.scales, result.scores, lw=1.5)
            plt.axvline(result.scale_optimal, color="r", linestyle="--", lw=1.5)
            plt.xlabel("pixel scale (nm/pixel)")
            plt.ylabel("correlation score")
            plt.show()
            _Logger.print_html(
                f"Optimal scale: <b>{result.scale_optimal:.4f}</b> nm/pixel"
            )
        return

    def _apply_method(self, layer: Image, method_name: str, *args, **kwargs):
        img = _convert_array(layer.data, layer.scale[-1])
        with set_gpu():
            out = getattr(img, method_name)(*args, **kwargs)
        return (
            out,
            {
                "scale": layer.scale,
                "translate": layer.translate,
                "name": f"{layer.name}-{method_name}",
                "rendering": layer.rendering,
            },
            "image",
        )

    def _preview_context(self, old_layer: Image, data, kwargs: dict[str, Any]):
        kwargs["name"] = PREVIEW_LAYER_NAME
        if PREVIEW_LAYER_NAME in self._viewer.layers:
            layer: Image = self._viewer.layers[PREVIEW_LAYER_NAME]
            layer.data = data
        else:
            layer = self._viewer.add_layer(Image(data, **kwargs))
        old_layer.visible = False
        is_active = yield
        if not is_active and layer in self._viewer.layers:
            self._viewer.layers.remove(layer)
            old_layer.visible = True


@magicclass(record=False)
class PlaneClip(MagicTemplate):
    layer = vfield(Layer)
    x = vfield(tuple[float, float], widget_type=FloatRangeSlider)
    y = vfield(tuple[float, float], widget_type=FloatRangeSlider)
    z = vfield(tuple[float, float], widget_type=FloatRangeSlider)

    @property
    def xmin_plane(self):
        return self.layer.experimental_clipping_planes[0]

    @property
    def xmax_plane(self):
        return self.layer.experimental_clipping_planes[1]

    @property
    def ymin_plane(self):
        return self.layer.experimental_clipping_planes[2]

    @property
    def ymax_plane(self):
        return self.layer.experimental_clipping_planes[3]

    @property
    def zmin_plane(self):
        return self.layer.experimental_clipping_planes[4]

    @property
    def zmax_plane(self):
        return self.layer.experimental_clipping_planes[5]

    @x.connect
    def _update_x(self):
        xmin, xmax = self.x
        with suppress(IndexError):
            self.xmin_plane.position = (0,) * (self.layer.ndim - 1) + (xmin,)
            self.xmax_plane.position = (0,) * (self.layer.ndim - 1) + (xmax,)
        return None

    @y.connect
    def _update_y(self):
        ymin, ymax = self.y
        with suppress(IndexError):
            self.ymin_plane.position = (0,) * (self.layer.ndim - 2) + (ymin, 0)
            self.ymax_plane.position = (0,) * (self.layer.ndim - 2) + (ymax, 0)
        return None

    @z.connect
    def _update_z(self):
        zmin, zmax = self.z
        with suppress(IndexError):
            self.zmin_plane.position = (0,) * (self.layer.ndim - 3) + (zmin, 0, 0)
            self.zmax_plane.position = (0,) * (self.layer.ndim - 3) + (zmax, 0, 0)
        return None

    @layer.connect
    def _connect_layer(self):
        layer = self.layer
        if not layer:
            return
        xmin = layer.extent.world[0, -1]
        xmax = layer.extent.world[1, -1]
        ymin = layer.extent.world[0, -2]
        ymax = layer.extent.world[1, -2]
        zmin = layer.extent.world[0, -3]
        zmax = layer.extent.world[1, -3]

        self["x"].range = xmin, xmax
        self["y"].range = ymin, ymax
        self["z"].range = zmin, zmax

        if len(self.layer.experimental_clipping_planes) == 6:
            self.x = self.xmin_plane.position[2], self.xmax_plane.position[2]
            self.x = self.ymin_plane.position[1], self.ymax_plane.position[1]
            self.x = self.zmin_plane.position[0], self.zmax_plane.position[0]

        else:
            self.x = xmin, xmax
            self.y = ymin, ymax
            self.z = zmin, zmax
            ndim = layer.ndim
            if ndim < 3:
                return
            extra_dim = (0,) * (ndim - 3)

            self.layer.experimental_clipping_planes = [
                {
                    "position": (0,) * (ndim - 1) + (xmin,),
                    "normal": extra_dim + (0, 0, 1),
                    "enabled": True,
                },
                {
                    "position": (0,) * (ndim - 1) + (xmax,),
                    "normal": extra_dim + (0, 0, -1),
                    "enabled": True,
                },
                {
                    "position": (0,) * (ndim - 2) + (ymin, 0),
                    "normal": extra_dim + (0, 1, 0),
                    "enabled": True,
                },
                {
                    "position": (0,) * (ndim - 2) + (ymax, 0),
                    "normal": extra_dim + (0, -1, 0),
                    "enabled": True,
                },
                {
                    "position": (0,) * (ndim - 3) + (zmin, 0, 0),
                    "normal": extra_dim + (1, 0, 0),
                    "enabled": True,
                },
                {
                    "position": (0,) * (ndim - 3) + (zmax, 0, 0),
                    "normal": extra_dim + (-1, 0, 0),
                    "enabled": True,
                },
            ]
