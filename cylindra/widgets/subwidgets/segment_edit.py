from typing import TYPE_CHECKING

import numpy as np
from magicclass import magicclass, vfield

from cylindra.components import CylSpline
from cylindra.widgets._widget_ext import JsonValueEdit
from cylindra.widgets.subwidgets._child_widget import ChildWidget

if TYPE_CHECKING:
    from napari.layers import Points


@magicclass(record=False)
class SplineSegmentEdit(ChildWidget):
    segment_value = vfield(JsonValueEdit)
    activate_3d_pick = vfield(False, label="Activate 3D pick")
    pick_max_distance = vfield(20.0, label="Max pick distance (nm)")
    interval = vfield(1.0, label="Sampling interval (nm)")

    def __post_init__(self):
        self._default_size = 8.0

    @property
    def spline(self) -> int | None:
        main = self._get_main()
        return main.SplineControl.num

    @activate_3d_pick.connect
    def _on_activate_3d_pick_changed(self, activated: bool):
        main = self._get_main()
        layer = main._reserved_layers.work
        if activated:
            self._default_size = float(layer.current_size)
            layer.current_size = 12.0
            layer.mouse_drag_callbacks.append(self._draw_callback)
            layer.mode = "pan_zoom"
        else:
            layer.mouse_drag_callbacks.remove(self._draw_callback)
            layer.mode = "add"
            layer.current_size = self._default_size

    def _draw_callback(self, layer: "Points", event):
        # See https://melissawm.github.io/napari-docs/gallery/cursor_ray.html
        # to know how to get 3D ray from 2D mouse event.
        if (i := self.spline) is None:
            return
        mouse_pos_start = np.asarray(event.position)
        yield
        while event.type == "mouse_move":
            yield
        # Mouse released here. Retrieve the ray and find the closest point on spline.
        mouse_pos_end = np.asarray(event.position)
        if np.linalg.norm(mouse_pos_end - mouse_pos_start) > 2.0:
            return  # not a click event

        main = self._get_main()
        near_point, far_point = main._reserved_layers.image.get_ray_intersections(
            mouse_pos_end, np.asarray(event.view_direction), event.dims_displayed
        )
        scale = main._reserved_layers.image.scale
        if near_point is None or far_point is None:
            return  # not intersecting the layer

        ray = CylSpline.line(near_point * scale, far_point * scale)

        # find the closest point on the spline
        spl = main.splines[i]
        u = np.linspace(0, 1, 256)
        ray_samples = ray.map(u)
        dist = spl.distance_matrix(ray_samples, interval=self.interval)
        min_idx = dist.argmin()
        if dist[min_idx] > self.pick_max_distance:
            return  # too far
        p_spl = dist.spl_points[min_idx[0]]
        layer.add(p_spl)

    def move_forward(self):
        """Move the selected point forward along the spline."""
        self._move_point(self.interval)

    def move_backward(self):
        """Move the selected point backward along the spline."""
        self._move_point(-self.interval)

    def _move_point(self, diff: float):
        if self.spline is None:
            return
        main = self._get_main()
        spl = main.splines[self.spline]
        layer = main._reserved_layers.work
        if len(sels := layer.selected_data) != 1:
            raise ValueError("Please select exactly one point to move.")
        data_old = layer.data
        data_index = list(sels)[0]
        point = data_old[data_index]
        dist = spl.distance_matrix(point, interval=self.interval)
        min_idx = np.argmin(dist.matrix, axis=0)
        if dist.matrix[min_idx[0], 0] > self.pick_max_distance:
            raise ValueError("The selected point is too far from the spline.")
        u0 = dist.spl_coords[min_idx[0]]
        u1 = max(min(u0 + diff / spl.length(), 1.0), 0.0)
        data_old[data_index] = spl.map(u1)
        layer.data = data_old
        layer.selected_data = [data_index]

    def add_segment(self):
        main = self._get_main()
        if self.spline is None:
            return
        spl = main.splines[self.spline]
        layer = main._reserved_layers.work
        if layer.data.shape[0] != 2:
            raise ValueError("Please add exactly two points to define the segment.")
        dist = spl.distance_matrix(layer.data, interval=self.interval)
        min_idx = np.argmin(dist.matrix, axis=0)
        u0 = dist.spl_coords[min_idx[0]]
        u1 = dist.spl_coords[min_idx[1]]
        if u0 > u1:
            u0, u1 = u1, u0
        y0, y1 = spl.distances([u0, u1])
        main.add_segment(self.spline, y0, y1, self.segment_value)
        layer.data = []  # clear points
