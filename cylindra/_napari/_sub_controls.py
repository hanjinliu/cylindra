from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from fnmatch import fnmatch
from typing import TYPE_CHECKING

import numpy as np
from magicclass.ext.polars import DataFrameView
from napari._qt.layer_controls.widgets import QtWidgetControlsBase
from napari._qt.layer_controls.widgets.qt_widget_controls_base import QtWrappedLabel
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from qtpy import QtCore
from qtpy import QtWidgets as QtW
from qtpy.QtCore import Qt
from superqt import QEnumComboBox, QLabeledDoubleSlider

from cylindra._napari._layers import (
    InteractionVector,
    LandscapeSurface,
    MoleculesLayer,
    SplineLayer,
)
from cylindra.utils import roundint

if TYPE_CHECKING:
    import pandas as pd


@contextmanager
def qt_signals_blocked(obj: QtCore.QObject):
    previous = obj.blockSignals(True)
    try:
        yield
    finally:
        obj.blockSignals(previous)


class ViewDimension(Enum):
    """Enum for how to display the molecules."""

    flat = 2
    sphere = 3


class QtFaceControls(QtWidgetControlsBase):
    _layer: MoleculesLayer

    def __init__(
        self,
        parent: QtW.QWidget,
        layer: MoleculesLayer,
        tooltip: str | None = None,
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.face_color.connect(self._on_face_color_change)

        # Setup widgets
        self.face_color_edit = QColorSwatchEdit(
            initial_color=self._layer.current_face_color,
            tooltip=tooltip,
        )
        self.face_color_label = QtWrappedLabel("face color:")
        self._on_face_color_change()
        self.face_color_edit.color_changed.connect(self._on_face_color_edit_changed)

    def _on_face_color_change(self) -> None:
        """Receive layer model face color change event and update color swatch."""
        with qt_signals_blocked(self.face_color_edit):
            if isinstance(col := self._layer._colormap_info, str):
                self.face_color_edit.setColor(col)
            else:
                nstops = 6
                colors = col.cmap.map(np.linspace(0, 1, nstops))
                # Build a horizontal gradient from the colormap colors
                rgba_stops: list[str] = []
                for i, c in enumerate(colors * 255):
                    r, g, b, _ = c
                    rgba = f"rgb({roundint(r)}, {roundint(g)}, {roundint(b)})"
                    pos = i / max(len(colors) - 1, 1)
                    rgba_stops.append(f"stop:{pos:.3f} {rgba}")
                gradient = ", ".join(rgba_stops)
                self.face_color_edit.color_swatch.setStyleSheet(
                    f"#colorSwatch {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {gradient}); }}"
                )

    def _on_face_color_edit_changed(self, color) -> None:
        self._layer.face_color = color

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QtW.QWidget]]:
        return [(self.face_color_label, self.face_color_edit)]


class QtBorderControls(QtWidgetControlsBase):
    _layer: MoleculesLayer

    def __init__(self, parent: QtW.QWidget, layer: MoleculesLayer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self.border_color_edit = QColorSwatchEdit(
            initial_color=self._layer.current_border_color,
            tooltip="Click to set the border color of currently selected points and any added afterwards.",
        )
        self.border_color_edit.color_changed.connect(self._change_border_color)
        layer.events.border_color.connect(self._on_border_color_change)

        self.border_width_edit = QtW.QDoubleSpinBox()
        self.border_width_edit.setRange(0.0, 1.0)
        self.border_width_edit.valueChanged.connect(self._change_border_width)
        self.border_width_edit.setSingleStep(0.05)
        self.border_width_edit.setValue(_first_or(layer.border_width, 0.05))
        self.border_width_edit.setToolTip("Width of the molecule borders")
        layer.events.border_width.connect(self._on_border_width_change)

        self.border_color_edit_label = QtWrappedLabel("border color:")
        self.border_width_label = QtWrappedLabel("border width:")

    def _on_border_color_change(self, event) -> None:
        """Receive layer.current_border_color() change event and update view."""
        with qt_signals_blocked(self.border_color_edit):
            self.border_color_edit.setColor(
                _first_or(self._layer.border_color, "dimgray")
            )

    def _on_border_width_change(self, event):
        with qt_signals_blocked(self.border_width_edit):
            self.border_width_edit.setValue(_first_or(self._layer.border_width, 0.05))

    def _change_border_width(self, value: float):
        self._layer.border_width = value

    def _change_border_color(self, value):
        self._layer.border_color = value

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QtW.QWidget]]:
        return [
            (self.border_color_edit_label, self.border_color_edit),
            (self.border_width_label, self.border_width_edit),
        ]


class QtPointStateControl(QtWidgetControlsBase):
    _layer: MoleculesLayer

    def __init__(self, parent: QtW.QWidget, layer: MoleculesLayer) -> None:
        super().__init__(parent, layer)
        # Point size slider
        self.point_size_slider = QLabeledDoubleSlider(
            orientation=Qt.Orientation.Horizontal
        )
        self.point_size_slider.setRange(0.1, 12.0)
        self.point_size_slider.setSingleStep(0.02)
        self.point_size_slider.setValue(layer.point_size)
        self.point_size_slider.setToolTip("Point size of all the molecules")
        self.point_size_slider.valueChanged.connect(self._change_point_size)
        layer.events.point_size.connect(self._on_point_size_change)

        # View dimension combo box
        self.dimComboBox = QEnumComboBox(enum_class=ViewDimension)
        self.dimComboBox.currentEnumChanged.connect(self._change_dim)
        self.dimComboBox.setCurrentEnum(ViewDimension(layer.view_ndim))
        layer.events.view_ndim.connect(self._on_dim_change)

        self.property_filter = QtW.QLineEdit()
        self.property_filter.setPlaceholderText("Filter status tip ...")
        self.property_filter.setToolTip(
            "Filter the mouse-hover status tip by the column name.\n"
            "e.g. 'align-* will match all columns starting with 'align-'\n"
        )
        self.property_filter.editingFinished.connect(self._set_property_filter)

        self.point_size_label = QtWrappedLabel("point size:")
        self.dim_label = QtWrappedLabel("rendering:")
        self.property_filter_label = QtWrappedLabel("filter status:")

    def _on_point_size_change(self, event):
        with qt_signals_blocked(self.point_size_slider):
            self.point_size_slider.setValue(event.value)

    def _on_dim_change(self, event):
        with qt_signals_blocked(self.dimComboBox):
            self.dimComboBox.setCurrentEnum(ViewDimension(event.value))

    def _change_point_size(self, value: float):
        self._layer.point_size = value

    def _change_dim(self, value: ViewDimension):
        self._layer.view_ndim = value.value

    def _set_property_filter(self):
        text = self.property_filter.text()
        if text:
            self._layer._property_filter = lambda x: fnmatch(x, text)
        else:
            self._layer._property_filter = lambda _: True

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QtW.QWidget]]:
        return [
            (self.point_size_label, self.point_size_slider),
            (self.dim_label, self.dimComboBox),
            (self.property_filter_label, self.property_filter),
        ]


class QtHasFeaturesControls(QtWidgetControlsBase):
    _layer: MoleculesLayer | InteractionVector

    def __init__(self, parent: QtW.QWidget, layer: MoleculesLayer) -> None:
        super().__init__(parent, layer)
        self.feature_btns = QtW.QWidget()
        btns_layout = QtW.QHBoxLayout(self.feature_btns)
        btns_layout.setContentsMargins(0, 0, 0, 0)
        btns_layout.setSpacing(1)
        self.show_feature_btn = QtW.QPushButton("show", parent)
        self.show_feature_btn.setToolTip("Show features of the molecules in a table")
        self.show_feature_btn.clicked.connect(self._show_features)
        btns_layout.addWidget(self.show_feature_btn)
        self.copy_feature_btn = QtW.QPushButton("copy", parent)
        self.copy_feature_btn.setToolTip("Copy features of the molecules to clipboard")
        self.copy_feature_btn.clicked.connect(self._copy_features)
        btns_layout.addWidget(self.copy_feature_btn)

        self.feature_buttons_label = QtWrappedLabel("features:")

    def _show_features(self):
        from cylindra.widget_utils import show_widget

        if isinstance(self._layer, MoleculesLayer):
            df = self._layer.molecules.features
        else:
            df = self._layer.net.features
        table = DataFrameView(value=df)
        show_widget(table, f"Features of {self._layer.name!r}", self.parent())

    def _copy_features(self):
        df: pd.DataFrame = self._layer.features
        df.to_clipboard(index=False)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QtW.QWidget]]:
        return [(self.feature_buttons_label, self.feature_btns)]


class QtLandscapeSubControls(QtWidgetControlsBase):
    _layer: LandscapeSurface

    def __init__(self, parent: QtW.QWidget, layer: LandscapeSurface) -> None:
        super().__init__(parent, layer)
        self.levelSlider = QLabeledDoubleSlider(orientation=Qt.Orientation.Horizontal)
        self.levelSlider.setRange(layer._level_min, layer._level_max)
        self.level_label = QtWrappedLabel("level:")
        self.levelSlider.sliderReleased.connect(self._change_level)
        self.levelSlider.setSingleStep(0.001)
        self.levelSlider.setValue(layer.level)
        self.levelSlider.setToolTip("Threshold level for surface rendering")
        layer.events.level.connect(self._on_level_change)

        self.resolution = QtW.QDoubleSpinBox()
        self.resolution_label = QtWrappedLabel("resolution:")
        self.resolution.setRange(0.1, 10)
        self.resolution.setSingleStep(0.05)
        self.resolution.setToolTip("Resolution of the surface")
        self.resolution.setValue(layer.resolution)
        self.resolution.valueChanged.connect(self._change_resolution)
        layer.events.resolution.connect(self._on_resolution_change)

        self.wireWidth = QtW.QDoubleSpinBox()
        self.wireWidth.setRange(0.0, 10.0)
        self.wire_width_label = QtWrappedLabel("wire width:")
        self.wireWidth.valueChanged.connect(self._change_wire_width)
        self.wireWidth.setSingleStep(0.05)
        self.wireWidth.setValue(layer.wireframe.width)
        self.wireWidth.setToolTip("Width of the wireframe")
        layer.wireframe.events.width.connect(self._on_wire_width_change)

        self.showMinCheckBox = QtW.QCheckBox()
        self.show_min_label = QtWrappedLabel("show min:")
        self.showMinCheckBox.setChecked(layer.show_min)
        self.showMinCheckBox.checkStateChanged.connect(self._change_show_min)
        self.showMinCheckBox.setToolTip(
            "Show the surface of the minimum values of the energy"
        )
        layer.events.show_min.connect(self._on_show_min_change)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QtW.QWidget]]:
        return [
            (self.level_label, self.levelSlider),
            (self.resolution_label, self.resolution),
            (self.wire_width_label, self.wireWidth),
            (self.show_min_label, self.showMinCheckBox),
        ]

    def _on_level_change(self, event):
        with qt_signals_blocked(self.levelSlider):
            self.levelSlider.setValue(event.value)

    def _on_resolution_change(self, event):
        with qt_signals_blocked(self.resolution):
            self.resolution.setValue(event.value)

    def _on_show_min_change(self, event):
        with qt_signals_blocked(self.showMinCheckBox):
            self.showMinCheckBox.setChecked(bool(event.value))

    def _on_wire_width_change(self, event):
        with qt_signals_blocked(self.wireWidth):
            self.wireWidth.setValue(event.value)

    def _change_level(self):
        self._layer.level = self.levelSlider.value()

    def _change_resolution(self):
        self._layer.resolution = self.resolution.value()

    def _change_show_min(self, state):
        self._layer.show_min = state == Qt.CheckState.Checked

    def _change_wire_width(self, value):
        self._layer.wireframe.width = value


class QtSplineLayerSubControl(QtWidgetControlsBase):
    _layer: SplineLayer

    def __init__(self, parent: QtW.QWidget, layer: SplineLayer) -> None:
        super().__init__(parent, layer)
        self.checkbox = QtW.QCheckBox()
        self.checkbox.setChecked(layer._show_polarity)
        layer.events.show_polarity.connect(self._on_show_polarity_change)
        self.checkbox.checkStateChanged.connect(self._change_show_polarity)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QtW.QWidget]]:
        return [(QtWrappedLabel("show orientation:"), self.checkbox)]

    def _on_show_polarity_change(self, event):
        with qt_signals_blocked(self.checkbox):
            self.checkbox.setChecked(bool(event.value))

    def _change_show_polarity(self, *_):
        self._layer.show_polarity = self.checkbox.isChecked()


def _first_or(arr: np.ndarray, default):
    """Get the first element of the array or the default value."""
    if arr.size > 0:
        return arr[0]
    return default
