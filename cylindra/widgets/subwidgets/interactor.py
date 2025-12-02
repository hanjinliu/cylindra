from collections import defaultdict
from typing import Annotated, Iterator

import numpy as np
from magicclass import abstractapi, magicclass, set_design, vfield

from cylindra.components import CylSpline
from cylindra.widgets._widget_ext import JsonValueEdit
from cylindra.widgets.subwidgets._child_widget import ChildWidget

DISABLED = "Disabled"
SELECT_SPLINE = "Select spline"
ADD_POINT_ON_SPLINE = "Add point on spline"

_PICK_MODES = [
    DISABLED,
    SELECT_SPLINE,
    ADD_POINT_ON_SPLINE,
]


@magicclass(record=False)
class Spline3DInteractor(ChildWidget):
    pick_mode_left = vfield(DISABLED).with_choices(_PICK_MODES)
    pick_mode_right = vfield(DISABLED).with_choices(_PICK_MODES)
    pick_max_distance = vfield(10.0, label="Max pick distance (nm)")
    interval = vfield(1.0, label="Precision (nm)").with_options(step=0.1, min=0.2)

    @magicclass(layout="horizontal")
    class MovePanel(ChildWidget):
        label0 = vfield("Move:", widget_type="Label")
        move_backward = abstractapi()
        move_forward = abstractapi()

    def __init__(self):
        self._pick_func_left = self._no_action
        self._pick_func_right = self._no_action

    def _init(self):
        self._add_mouse_callback()
        self.pick_mode_left = SELECT_SPLINE
        self.pick_mode_right = ADD_POINT_ON_SPLINE

    def _add_mouse_callback(self):
        main = self._get_main()
        if self._draw_callback not in main._reserved_layers.work.mouse_drag_callbacks:
            main._reserved_layers.work.mouse_drag_callbacks.append(self._draw_callback)

    def _iter_splines(self) -> Iterator[tuple[int, CylSpline]]:
        main = self._get_main()
        out = list(main.splines.enumerate())
        if i := main.SplineControl.num is not None:
            out = [(i, main.splines[i])] + [a for a in out if a[0] != i]
        yield from out

    @pick_mode_left.connect
    def _on_pick_mode_changed(self, mode: str):
        if mode == ADD_POINT_ON_SPLINE:
            self._pick_func_left = self._add_point_on_spline
        elif mode == SELECT_SPLINE:
            self._pick_func_left = self._select_spline
        else:
            self._pick_func_left = self._no_action

    @pick_mode_right.connect
    def _on_pick_mode_right_changed(self, mode: str):
        if mode == ADD_POINT_ON_SPLINE:
            self._pick_func_right = self._add_point_on_spline
        elif mode == SELECT_SPLINE:
            self._pick_func_right = self._select_spline
        else:
            self._pick_func_right = self._no_action

    def _draw_callback(self, layer, event):
        # See https://melissawm.github.io/napari-docs/gallery/cursor_ray.html
        # to know how to get 3D ray from 2D mouse event.
        if event.modifiers or len(event.dims_displayed) == 2:
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
        if near_point is None or far_point is None:
            return  # not intersecting the layer
        # button 1: left, 2: right
        if event.button == 1:
            self._pick_func_left(near_point, far_point)
        elif event.button == 2:
            self._pick_func_right(near_point, far_point)

    def _no_action(self, near_point: np.ndarray, far_point: np.ndarray):
        pass

    def _select_spline(self, near_point: np.ndarray, far_point: np.ndarray):
        main = self._get_main()
        scale = main._reserved_layers.image.scale

        for idx, spl in main.splines.enumerate():
            p_spl, d0 = _closest_points(
                spl,
                near_point * scale,
                far_point * scale,
                interval=self.interval,
            )
            if d0 <= self.pick_max_distance:
                main.SplineControl.num = idx
                if spl.has_anchors:
                    # move slider to the closest anchor
                    dists = np.linalg.norm(
                        spl.map(spl.anchors) - p_spl.reshape(1, 3), axis=1
                    )
                    main.SplineControl.pos = np.argmin(dists)
                return

    def _add_point_on_spline(self, near_point: np.ndarray, far_point: np.ndarray):
        main = self._get_main()
        scale = main._reserved_layers.image.scale

        for _, spl in self._iter_splines():
            p_spl, d0 = _closest_points(
                spl,
                near_point * scale,
                far_point * scale,
                interval=self.interval,
            )
            if d0 <= self.pick_max_distance:
                main._reserved_layers.work.add(p_spl)
                size = main._reserved_layers.work.size
                size[-1] = 12.0
                main._reserved_layers.work.size = size
                return

    @set_design(text="Forward", location=MovePanel)
    def move_forward(self, interval: Annotated[float, {"bind": interval}]):
        """Move the selected point forward along the spline."""
        self._move_point(interval)

    @set_design(text="Backward", location=MovePanel)
    def move_backward(self, interval: Annotated[float, {"bind": interval}]):
        """Move the selected point backward along the spline."""
        self._move_point(-interval)

    def _move_point(self, diff: float):
        main = self._get_main()
        layer = main._reserved_layers.work
        if len(sels := layer.selected_data) > 1:
            raise ValueError("Please select exactly one point to move.")
        elif len(sels) == 0:
            if layer.data.size == 0:
                return
            sels = {0}
        data_old = layer.data
        data_index = list(sels)[0]
        point = data_old[data_index]
        for _, spl in self._iter_splines():
            dist = spl.distance_matrix(point, interval=self.interval)
            min_idx = np.argmin(dist.matrix, axis=0)
            if dist.matrix[min_idx[0], 0] <= self.pick_max_distance:
                break
        else:
            raise ValueError("No nearest spline found.")
        u0 = dist.spl_coords[min_idx[0]]
        u1 = max(min(u0 + diff / spl.length(), 1.0), 0.0)
        data_old[data_index] = spl.map(u1)
        layer.data = data_old
        layer.selected_data = [data_index]

    @set_design(text="Add segment")
    def add_segment(self):
        """Add a segment between two points."""
        main = self._get_main()
        idx, (y0, y1) = self._determine_spline_and_points()
        main.add_segment(idx, y0, y1, self.segment_value)
        main._reserved_layers.work.data = []  # clear points

    segment_value = vfield(JsonValueEdit)

    @set_design(text="Delete segments")
    def delete_segments(self):
        """Delete all the segments near the added points."""
        main = self._get_main()
        layer = main._reserved_layers.work
        points = layer.data
        arg_map = defaultdict[int, list[int]](list)
        for idx, spl in self._iter_splines():
            for i_seg, seg in spl.segments.enumerate():
                spl_seg = spl.clip(seg.start, seg.end)
                dist = spl_seg.distance_matrix(points, interval=self.interval)
                if np.min(dist.matrix) <= self.pick_max_distance:
                    arg_map[idx].append(i_seg)
        for spl_idx, seg_indices in arg_map.items():
            main.delete_segments(spl_idx, seg_indices)
        layer.data = []  # clear points

    @set_design(text="Clip spline")
    def clip_spline(self):
        """Clip spline between the added two points."""
        main = self._get_main()
        idx, (y0, y1) = self._determine_spline_and_points()
        main.clip_spline(idx, (y0, main.splines[idx].length() - y1))
        main._reserved_layers.work.data = []  # clear points

    @set_design(text="Split spline")
    def split_spline(self):
        """Split spline at the added point."""
        main = self._get_main()
        idx, (y0,) = self._determine_spline_and_points(num=1)
        main.split_spline(idx, y0, trim=self.trim)
        main._reserved_layers.work.data = []  # clear points

    trim = vfield(0.0, label="Trim when splitting (nm)").with_options(
        step=0.1, min=0.0, max=100.0
    )

    def _determine_spline_and_points(self, num: int = 2) -> tuple[int, list[float]]:
        main = self._get_main()
        layer = main._reserved_layers.work
        points = layer.data
        if points.shape[0] != num:
            raise ValueError(f"Please add exactly {num} points.")

        for _idx, spl in self._iter_splines():
            dist = spl.distance_matrix(points, interval=self.interval)
            min_idx = np.argmin(dist.matrix, axis=0)
            if np.all(
                dist.matrix[min_idx, np.arange(num, dtype=np.int32)]
                <= self.pick_max_distance
            ):
                break
        else:
            raise ValueError("No nearest spline found.")
        us = np.sort(dist.spl_coords[min_idx])
        return _idx, [float(y) for y in spl.distances(us).round(3)]


def _closest_points(
    spl: CylSpline,
    near_point_nm: np.ndarray,
    far_point_nm: np.ndarray,
    interval: float,
) -> tuple[np.ndarray, float]:
    # Use point-to-line distance formula
    # For a line through points A and B, distance from point P is:
    # d = ||(P - A) × (B - A)|| / ||B - A||

    line_vec = far_point_nm - near_point_nm
    line_length = np.linalg.norm(line_vec)
    line_vec_normalized = line_vec / line_length

    # Sample points along the spline
    u = spl.prep_anchor_positions(max_interval=interval)
    spl_points = spl.map(u)

    # Calculate distances from each spline point to the line
    vecs = spl_points - near_point_nm
    cross_products = np.cross(vecs, line_vec_normalized)
    distances = np.linalg.norm(cross_products, axis=1)

    min_idx = np.argmin(distances)
    d0 = distances[min_idx]
    p_spl = spl_points[min_idx]

    return p_spl, d0
