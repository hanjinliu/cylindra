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


@magicclass(record=False)
class ManualPicker(ChildWidget):
    @magictoolbar(record=False)
    class Toolbar(ChildWidget):
        register = abstractapi()
        undo_last = abstractapi()
        clear_all = abstractapi()
        refresh_widget_state = abstractapi()

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

        def _get_available_binsize(self, widget=None) -> "list[int]":
            try:
                return self._get_main()._get_available_binsize(widget)
            except Exception:
                return []

        depth = vfield(4.0, label="depth (nm)").with_options(min=1.0, max=200.0, step=1.0)  # fmt: skip
        width_ = vfield(80.0, label="width (nm)").with_options(min=10.0, max=300.0, step=1.0)  # fmt: skip
        binsize = vfield().with_choices(_get_available_binsize)

    @magicclass(layout="horizontal", widget_type="split")
    class Row0(ChildWidget):
        canvas = abstractapi()

        @magicclass(widget_type="frame", labels=False)
        class image_params(ChildWidget):
            enable_lowpass = vfield(ToggleSwitch, label="Low-pass filter")
            label_cutoff = field("Cutoff (nm)", widget_type="Label")
            lowpass_cutoff = field(2.0).with_options(
                min=0.1, max=10.0, step=0.1, enabled=False
            )

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
        pos = field(float, label="Position (nm)", widget_type="FloatSlider").with_options(max=0)  # fmt: skip

    @magicclass(widget_type="frame")
    class Rot(ChildWidget):
        """Rotation of the plane along the spline."""

        roll = vfield(0.0, widget_type="FloatSlider", label="Roll angle (°)").with_options(min=-180, max=180, step=1)  # fmt: skip
        pitch = vfield(0.0, widget_type="FloatSlider", label="Pitch angle (°)").with_options(min=-90, max=90, step=1)  # fmt: skip
        yaw = vfield(0.0, widget_type="FloatSlider", label="Yaw angle (°)").with_options(min=-90, max=90, step=1)  # fmt: skip
        focus = vfield(0.0, widget_type="FloatSlider", label="Focus offset (nm)").with_options(min=-100, max=100, step=1)  # fmt: skip

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
        return None

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

        return None

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

    def _calc_image_slice(self):
        tomo = self._get_main().tomogram
        binsize = self.params.binsize

        # Calculate the coordinates
        mole = self._molecule_at_pos()
        scale = tomo.scale * binsize
        d_px = roundint(self.params.depth / scale)
        w_px = roundint(self.params.width_ / scale)
        coords = mole.local_coordinates((w_px, d_px, w_px), scale)

        # trasform image
        img = tomo._get_multiscale_or_original(self.params.binsize)
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

        if self.Row0.image_params.enable_lowpass:
            scale = main.tomogram.scale * self.params.binsize
            cutoff = self.Row0.image_params.lowpass_cutoff.value / scale
            cutoff_rel = 0.5 / cutoff
            img = img.lowpass_filter(cutoff_rel)

        @thread_worker.callback
        def _update_image():
            self.canvas.image = img
            self.canvas.text_overlay.visible = False
            factor = self.params._old_binsize / self.params.binsize
            if factor != 1:
                xlim = [(v + 0.5) * factor - 0.5 for v in self.canvas.xlim]
                ylim = [(v + 0.5) * factor - 0.5 for v in self.canvas.ylim]
                self.canvas.xlim = xlim
                self.canvas.ylim = ylim
            if update_clim or self.Row0.image_params.enable_autocontrast:
                img_min, img_max = img.min(), img.max()
                eps = 0.01 * (img_max - img_min)
                self.canvas.contrast_limits = img_min + eps, img_max - eps
            self.params._old_binsize = self.params.binsize

            # update point slices
            xdata, zdata, radius = self._points_in_canvas(img, mole)
            self._layer_points.data = xdata, zdata
            self._layer_points.size = radius * 2

            # update plane layer
            wy, wx, _o = img.shape[0] + 0.5, img.shape[1] + 0.5, -0.5
            points = [[_o, _o], [_o, wx], [wy, wx], [wy, _o], [_o, _o]]
            main._reserved_layers.plane.data = [
                self._plane_pos_to_world_pos(y0, x0) for x0, y0 in points
            ]

        return _update_image

    def _points_in_canvas(self, img: np.ndarray, mole: "Molecules"):
        main = self._get_main()
        scale = main.tomogram.scale * self.params.binsize
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

    @Rot.roll.connect_async(timeout=0.1)
    @Rot.pitch.connect_async(timeout=0.1)
    @Rot.yaw.connect_async(timeout=0.1)
    @Rot.focus.connect_async(timeout=0.1)
    @params.depth.connect_async(timeout=0.1)
    @params.width_.connect_async(timeout=0.1)
    @params.binsize.connect_async(timeout=0.1)
    @Row0.image_params.enable_lowpass.connect_async(timeout=0.1)
    @Row0.image_params.lowpass_cutoff.connect_async(timeout=0.1)
    @Row0.image_params.enable_autocontrast.connect_async(timeout=0.1)
    @controller.spline_id.connect_async(timeout=0.1)
    @controller.pos.connect_async(timeout=0.1, abort_limit=0.5)
    def _on_widget_state_changed(self):
        if self.visible:
            yield from self._update_canvas.arun()
        return None

    def _plane_pos_to_world_pos(self, y: float, x: float):
        cy, cx = (np.array(self.canvas.image.shape) - 1) / 2
        scale = self._get_main().tomogram.scale * self.params.binsize
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
        world_pos = self._plane_pos_to_world_pos(y, x)
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
        return None

    def _update_points_in_canvas(self):
        if self.canvas.image is None:
            return
        xdata, zdata, radius = self._points_in_canvas(
            self.canvas.image,
            self._molecule_at_pos(),
        )
        self._layer_points.data = xdata, zdata
        self._layer_points.size = radius * 2
        return None

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
        self.Rot.yaw = max(-90, self.Rot.yaw - 5)

    @bind_key("D")
    def _yaw_right(self):
        self.Rot.yaw = min(90, self.Rot.yaw + 5)

    @bind_key("W")
    def _pitch_up(self):
        self.Rot.pitch = min(90, self.Rot.pitch + 5)

    @bind_key("S")
    def _pitch_down(self):
        self.Rot.pitch = max(-90, self.Rot.pitch - 5)

    @bind_key("Q")
    def _roll_left(self):
        self.Rot.roll = max(-180, self.Rot.roll - 5)

    @bind_key("E")
    def _roll_right(self):
        self.Rot.roll = min(180, self.Rot.roll + 5)

    @bind_key("F")
    def _move_forward(self):
        self.controller.pos.value = min(
            self.controller.pos.max, self.controller.pos.value + 10
        )

    @bind_key("B")
    def _move_backward(self):
        self.controller.pos.value = max(0, self.controller.pos.value - 10)

    @bind_key("J")
    def _move_focus_up(self):
        self.Rot.focus = min(100, self.Rot.focus + 10)

    @bind_key("K")
    def _move_focus_down(self):
        self.Rot.focus = max(-100, self.Rot.focus - 10)

    @bind_key("Delete")
    @bind_key("Backspace")
    def _delete(self):
        main = self._get_main()
        main._reserved_layers.work.remove_selected()
        self._update_points_in_canvas()
