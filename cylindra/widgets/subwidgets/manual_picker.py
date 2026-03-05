from typing import TYPE_CHECKING

import impy as ip
import numpy as np
from magicclass import (
    abstractapi,
    bind_key,
    field,
    magicclass,
    magictoolbar,
    set_design,
    vfield,
)
from magicclass.ext.pyqtgraph import QtImageCanvas
from magicclass.ext.vispy import Vispy3DCanvas
from magicclass.utils import thread_worker
from magicclass.widgets import ToggleSwitch

from cylindra.utils import map_coordinates, roundint
from cylindra.widgets._main_utils import degrees_to_rotator
from cylindra.widgets.subwidgets._child_widget import ChildWidget

if TYPE_CHECKING:
    from acryo import Molecules
    from magicclass.ext.pyqtgraph.mouse_event import MouseClickEvent

_FOCUS_MIN = -100
_FOCUS_MAX = 100
_TIMEOUT = 0.05


@magicclass(record=False)
class ManualPicker(ChildWidget):
    @magictoolbar(record=False)
    class Toolbar(ChildWidget):
        """Toolbar

        Attributes
        ----------
        dist_step : int
            The step size for moving forward/backward using keybindings.
        angle_step : int
            The step size for changing angles using keybindings.
        """

        register = abstractapi()
        undo_last = abstractapi()
        clear_all = abstractapi()
        refresh_widget_state = abstractapi()

        dist_step = field(10, label="Dist step (nm)", widget_type="SpinBox").with_options(min=1, max=100, step=1)  # fmt: skip
        angle_step = field(5, label="Angle step (°)", widget_type="SpinBox").with_options(min=1, max=45, step=1)  # fmt: skip

    @magicclass(layout="horizontal")
    class params(ChildWidget):
        """Sweeper parameters.

        Attributes
        ----------
        depth : float
            The depth of the projection along splines. For instance, depth=50.0 means that Y-projection will be calculated
            using subvolume of size L * 50.0 nm * L.
        binsize : int
            The size of the binning. For instance, binsize=2 means that the image will be binned by 2 before projection
            and/or Fourier transformation.
        """

        def __init__(self):
            self._old_binsize = 1

        def _get_available_binsize(self, widget=None) -> "list[tuple[str, int]]":
            """List up all available bin sizes (0 means using reference)."""
            try:
                bin_sizes = self._get_main()._get_available_binsize(widget)
            except Exception:
                bin_sizes = []
            return bin_sizes + [("Reference image", 0)]

        depth = vfield(4.0, label="depth (nm)").with_options(min=1.0, max=200.0, step=1.0)  # fmt: skip
        width_ = vfield(80.0, label="width (nm)").with_options(min=10.0, max=300.0, step=1.0)  # fmt: skip
        binsize = vfield().with_choices(_get_available_binsize)

    @magicclass(layout="horizontal", widget_type="split")
    class Row0(ChildWidget):
        canvas = abstractapi()

        @magicclass(widget_type="frame", labels=False)
        class image_params(ChildWidget):
            """Settings.

            Attributes
            ----------
            enable_lowpass : bool
                Whether to apply low-pass filter to the image slice.
            lowpass_cutoff : float
                Cutoff frequency for the low-pass filter in nm.
            enable_autocontrast : bool
                Whether to automatically adjust contrast limits to the image slice.
            preview_3d : Vispy3DCanvas
                Conceptual view of the slicing plane along the spline in 3D.
            show_in_viewer : bool
                Whether to show the slicing plane in the napari viewer.
            """

            enable_lowpass = vfield(ToggleSwitch, label="Low-pass filter")
            label_cutoff = field("Cutoff (nm)", widget_type="Label")
            lowpass_cutoff = field(2.0).with_options(min=0.1, max=10.0, step=0.1, enabled=False)  # fmt: skip

            @enable_lowpass.connect
            def _on_enable_lowpass_changed(self, value: bool):
                self.lowpass_cutoff.enabled = value

            enable_autocontrast = vfield(ToggleSwitch, label="Auto contrast")
            preview_3d = field(Vispy3DCanvas)
            show_in_viewer = vfield(ToggleSwitch, label="Show in viewer")

    canvas = field(QtImageCanvas, location=Row0).with_options(lock_contrast_limits=True)

    @magicclass(widget_type="frame")
    class controller(ChildWidget):
        """Control spline positions.

        Attributes
        ----------
        spline_id : int
            Current spline ID to analyze.
        pos : nm
            Position along the spline in nm.
        """

        def _get_spline_id(self, widget=None) -> "list[tuple[str, int]]":
            try:
                return self._get_main()._get_splines(widget)
            except Exception:
                return []

        spline_id = vfield(label="Spline").with_choices(_get_spline_id)
        pos = field(float, label="Position (nm)", widget_type="FloatSlider").with_options(step=0.1, max=0)  # fmt: skip

    @magicclass(widget_type="frame")
    class Rot(ChildWidget):
        """Rotation and focus offset of the plane along the spline."""

        roll = vfield(0.0, widget_type="FloatSlider", label="Roll angle (°)").with_options(min=-180, max=180, step=1)  # fmt: skip
        pitch = vfield(0.0, widget_type="FloatSlider", label="Pitch angle (°)").with_options(min=-90, max=90, step=1)  # fmt: skip
        yaw = vfield(0.0, widget_type="FloatSlider", label="Yaw angle (°)").with_options(min=-90, max=90, step=1)  # fmt: skip
        focus = vfield(0.0, widget_type="FloatSlider", label="Focus offset (nm)").with_options(min=_FOCUS_MIN, max=_FOCUS_MAX, step=1)  # fmt: skip

    def __post_init__(self):
        self.canvas.mouse_clicked.connect(self._on_mouse_clicked)
        self._layer_points = self.canvas.add_scatter(
            [],
            [],
            face_color=[0, 0, 0, 0],
            edge_color="lime",
            lw=2,
        )
        self.Row0.image_params.preview_3d.add_curve(
            [[0, -1.5, 0], [0, 1.5, 0]], color="lime", width=2
        )
        self.Row0.image_params.preview_3d.add_curve(
            [[0, 0, 0], [0, 0, 0.5]], color="cyan", width=1
        )
        self.Row0.image_params.preview_3d.add_curve(
            [[0, 0, 0], [0.5, 0, 0]], color="orange", width=1
        )
        self._layer_plane = self.Row0.image_params.preview_3d.add_curve(
            [[-1, 0, -1], [-1, 0, 1], [1, 0, 1], [1, 0, -1], [-1, 0, -1]],
            color="gray",
        )
        self.Row0.image_params.max_width = 230

    @set_design(icon="gg:add", location=Toolbar)
    def register(self):
        """Register current points as a molecules layer."""
        main = self._get_main()
        main.register_molecules(main._get_spline_coordinates())
        self._layer_points.data = [], []

    @set_design(icon="mynaui:delete", location=Toolbar)
    def undo_last(self):
        main = self._get_main()
        work_layer = main._reserved_layers.work
        npoints = work_layer.data.shape[0]
        if npoints > 0:
            work_layer.selected_data = {work_layer.data.shape[0] - 1}
            work_layer.remove_selected()
            self._update_points_in_canvas()

    @set_design(icon="solar:eraser-bold", location=Toolbar)
    def clear_all(self):
        """Clear all the points."""
        main = self._get_main()
        main._reserved_layers.work.data = []
        self._layer_points.data = [], []

    @set_design(icon="tabler:refresh", location=Toolbar)
    def refresh_widget_state(self):
        """Refresh widget state."""
        self._spline_changed(self.controller.spline_id)
        self._update_canvas()
        self._update_preview_3d()

    @Row0.image_params.show_in_viewer.connect
    def _show_in_viewer(self, value: bool):
        main = self._get_main()
        if main.parent_viewer is None:
            return None
        plane = main._reserved_layers.plane
        if not value:
            if plane in main.parent_viewer.layers:
                main.parent_viewer.layers.remove(plane)
            return None

        tomo = main.tomogram
        if len(tomo.splines) == 0:
            return None
        if plane not in main.parent_viewer.layers:
            main.parent_viewer.add_layer(plane)

    def _current_rotator(self):
        return degrees_to_rotator(
            [("y", self.Rot.roll), ("x", self.Rot.pitch), ("z", self.Rot.yaw)]
        )

    def _molecule_at_pos(self):
        idx = self.controller.spline_id
        spl = self._get_main().tomogram.splines[idx]
        pos = self.controller.pos.value / spl.length()
        rotvec = self._current_rotator().as_rotvec()
        return (
            spl.anchors_to_molecules(pos)
            .rotate_by_rotvec_internal(rotvec)
            .translate_internal([0, self.Rot.focus, 0])
        )

    def _calc_image_slice(self) -> tuple[ip.ImgArray, "Molecules"]:
        tomo = self._get_main().tomogram
        bsize = self.params.binsize
        if bsize == 0:
            img = self._get_main()._reserved_layers.image_data
            bsize = img.scale.x / tomo.scale
        else:
            img = tomo._get_multiscale_or_original(bsize)

        # Calculate the coordinates
        mole = self._molecule_at_pos()
        scale = tomo.scale * bsize
        d_px = roundint(self.params.depth / scale)
        w_px = roundint(self.params.width_ / scale)
        coords = mole.local_coordinates((w_px, d_px, w_px), scale)

        # trasform image
        try:
            out = map_coordinates(img, coords, order=1).mean(axis="y")[:, ::-1]
        except ValueError:
            # out of range
            out = np.zeros((w_px, w_px), dtype=img.dtype)
        out = ip.asarray(out, axes="yx")
        out.set_scale(img, unit=img.scale_unit)
        return out, mole

    @controller.spline_id.connect
    def _spline_changed(self, idx: int):
        try:
            spl = self._get_main().tomogram.splines[idx]
            self.controller.pos.max = max(spl.length(), 0)
        except Exception:
            pass

    @thread_worker(force_async=True)
    def _update_canvas(self, update_clim: bool = False):
        if self.controller.spline_id is None:
            return
        main = self._get_main()
        img, mole = self._calc_image_slice()
        bsize = self._true_bin_size()

        if self.Row0.image_params.enable_lowpass:
            scale = main.tomogram.scale * bsize
            cutoff = self.Row0.image_params.lowpass_cutoff.value / scale
            cutoff_rel = 0.5 / cutoff
            img = img.lowpass_filter(cutoff_rel)

        yield self._update_canvas_image.with_args(img)
        self.params._old_binsize = bsize
        if (factor := self.params._old_binsize / bsize) != 1:
            yield self._update_canvas_xy_lims.with_args(factor)
        if update_clim or self.Row0.image_params.enable_autocontrast:
            yield self._update_canvas_clim.with_args(img)
        xdata, zdata, radius = self._points_in_canvas(img, mole)
        yield self._update_canvas_slice_data.with_args(xdata, zdata, radius)
        wy, wx, _o = img.shape[0] + 0.5, img.shape[1] + 0.5, -0.5
        points = [[_o, _o], [_o, wx], [wy, wx], [wy, _o], [_o, _o]]
        plane_data = [
            self._plane_pos_to_world_pos(y0, x0, img.shape) for x0, y0 in points
        ]
        yield self._update_plane_in_viewer.with_args(plane_data)

    def _points_in_canvas(self, img: np.ndarray, mole: "Molecules"):
        main = self._get_main()
        bsize = self._true_bin_size()
        scale = main.tomogram.scale * bsize
        data_transformed = mole.rotator.inv().apply(
            main._reserved_layers.work.data - mole.pos
        )

        dy = data_transformed[:, 1]
        radius2 = main._reserved_layers.work.size**2 - dy**2
        mask = radius2 > 0
        radius = np.sqrt(radius2[mask]) / scale
        cz, cx = (np.array(img.shape) - 1) / 2
        zdata = data_transformed[mask, 0] / scale + cz
        xdata = -data_transformed[mask, 2] / scale + cx
        return xdata, zdata, radius

    @Rot.roll.connect_async(timeout=_TIMEOUT)
    @Rot.pitch.connect_async(timeout=_TIMEOUT)
    @Rot.yaw.connect_async(timeout=_TIMEOUT)
    @Rot.focus.connect_async(timeout=_TIMEOUT)
    @params.depth.connect_async(timeout=_TIMEOUT)
    @params.width_.connect_async(timeout=_TIMEOUT)
    @params.binsize.connect_async(timeout=_TIMEOUT)
    @Row0.image_params.enable_lowpass.connect_async(timeout=_TIMEOUT)
    @Row0.image_params.lowpass_cutoff.connect_async(timeout=_TIMEOUT)
    @Row0.image_params.enable_autocontrast.connect_async(timeout=_TIMEOUT)
    @controller.spline_id.connect_async(timeout=_TIMEOUT)
    @controller.pos.connect_async(timeout=_TIMEOUT, abort_limit=0.5)
    def _on_widget_state_changed(self):
        if self.visible:
            yield from self._update_canvas.arun()

    def _plane_pos_to_world_pos(self, y: float, x: float, shape: tuple[int, int]):
        cy, cx = (np.array(shape) - 1) / 2
        bsize = self._true_bin_size()
        scale = self._get_main().tomogram.scale * bsize
        vx, vy = -(x - cx) * scale, (y - cy) * scale
        mole = self._molecule_at_pos()
        world_pos = mole.x.ravel() * vx + mole.z.ravel() * vy + mole.pos.ravel()
        return world_pos

    def _on_mouse_clicked(self, event: "MouseClickEvent"):
        if self.canvas.image is None:
            return
        x, y = event.pos()
        return self._mouse_click_impl(x, y, event.modifiers())

    def _mouse_click_impl(self, x: float, y: float, modifiers: tuple):
        world_pos = self._plane_pos_to_world_pos(y, x, self.canvas.image.shape)
        layer_work = self._get_main()._reserved_layers.work
        if modifiers == ():
            layer_work.add(world_pos)
        elif "control" in modifiers:
            layer_work.data = np.concatenate(
                [layer_work.data[:-1], world_pos[np.newaxis]], axis=0
            )
        else:
            return
        self._update_points_in_canvas()

    def _update_points_in_canvas(self):
        if self.canvas.image is None:
            return
        xdata, zdata, radius = self._points_in_canvas(
            self.canvas.image,
            self._molecule_at_pos(),
        )
        self._layer_points.data = xdata, zdata
        self._layer_points.size = radius * 2

    def _true_bin_size(self) -> float:
        bsize = self.params.binsize
        if bsize == 0:
            main = self._get_main()
            bsize = main._reserved_layers.image_data.scale.x / main.tomogram.scale
        return bsize

    @Rot.roll.connect
    @Rot.pitch.connect
    @Rot.yaw.connect
    @Rot.focus.connect
    def _update_preview_3d(self):
        rot = self._current_rotator()
        p00 = [-1, 0, -1]
        p01 = [-1, 0, 1]
        p10 = [1, 0, -1]
        p11 = [1, 0, 1]
        points = np.array([p00, p01, p11, p10, p00, p11, p01, p10])
        dy = self.Rot.focus / 50 * rot.apply([0, 1, 0])
        self._layer_plane.data = rot.apply(points) + dy

    @bind_key("A")
    def _yaw_left(self):
        self.Rot.yaw = max(-90, self.Rot.yaw - self.Toolbar.angle_step.value)

    @bind_key("D")
    def _yaw_right(self):
        self.Rot.yaw = min(90, self.Rot.yaw + self.Toolbar.angle_step.value)

    @bind_key("W")
    def _pitch_up(self):
        self.Rot.pitch = min(90, self.Rot.pitch + self.Toolbar.angle_step.value)

    @bind_key("S")
    def _pitch_down(self):
        self.Rot.pitch = max(-90, self.Rot.pitch - self.Toolbar.angle_step.value)

    @bind_key("Q")
    def _roll_left(self):
        self.Rot.roll = max(-180, self.Rot.roll - self.Toolbar.angle_step.value)

    @bind_key("E")
    def _roll_right(self):
        self.Rot.roll = min(180, self.Rot.roll + self.Toolbar.angle_step.value)

    @bind_key("F")
    def _move_forward(self):
        self.controller.pos.value = min(
            self.controller.pos.max,
            self.controller.pos.value + self.Toolbar.dist_step.value,
        )

    @bind_key("B")
    def _move_backward(self):
        self.controller.pos.value = max(
            self.controller.pos.min,
            self.controller.pos.value - self.Toolbar.dist_step.value,
        )

    @bind_key("J")
    def _move_focus_up(self):
        self.Rot.focus = min(_FOCUS_MAX, self.Rot.focus + self.Toolbar.dist_step.value)

    @bind_key("K")
    def _move_focus_down(self):
        self.Rot.focus = max(_FOCUS_MIN, self.Rot.focus - self.Toolbar.dist_step.value)

    @bind_key("Delete")
    @bind_key("Backspace")
    def _delete(self):
        main = self._get_main()
        main._reserved_layers.work.remove_selected()
        self._update_points_in_canvas()

    @thread_worker.callback
    def _update_canvas_image(self, img: np.ndarray):
        self.canvas.image = img
        self.canvas.text_overlay.visible = False

    @thread_worker.callback
    def _update_canvas_xy_lims(self, factor: float):
        xlim = [(v + 0.5) * factor - 0.5 for v in self.canvas.xlim]
        ylim = [(v + 0.5) * factor - 0.5 for v in self.canvas.ylim]
        self.canvas.xlim = xlim
        self.canvas.ylim = ylim

    @thread_worker.callback
    def _update_canvas_clim(self, img: np.ndarray):
        img_min, img_max = img.min(), img.max()
        eps = 0.01 * (img_max - img_min)
        self.canvas.contrast_limits = img_min + eps, img_max - eps

    @thread_worker.callback
    def _update_canvas_slice_data(self, xdata, zdata, radius):
        self._layer_points.data = xdata, zdata
        self._layer_points.size = radius * 2

    @thread_worker.callback
    def _update_plane_in_viewer(self, data):
        self._get_main()._reserved_layers.plane.data = data
