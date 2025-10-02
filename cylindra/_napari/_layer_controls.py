from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from fnmatch import fnmatch
from typing import TYPE_CHECKING

import napari
import numpy as np
from magicclass.ext.polars import DataFrameView
from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.qt_surface_controls import QtSurfaceControls
from napari._qt.layer_controls.widgets import (
    QtOutSliceCheckBoxControl,
    QtProjectionModeControl,
    QtTextVisibilityControl,
    QtWidgetControlsBase,
)
from napari._qt.layer_controls.widgets._vectors import (
    QtEdgeColorFeatureControl,
    QtVectorStyleComboBoxControl,
    QtWidthSpinBoxControl,
)
from napari._qt.layer_controls.widgets.qt_widget_controls_base import QtWrappedLabel
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.base._base_constants import Mode as BaseMode
from napari.layers.points._points_constants import Mode as PointsMode
from napari.utils.events import disconnect_events
from qtpy import QtCore
from qtpy import QtWidgets as QtW
from qtpy.QtCore import Qt
from superqt import QEnumComboBox, QLabeledDoubleSlider

if TYPE_CHECKING:
    import pandas as pd
    from qtpy.QtWidgets import QFormLayout

    from cylindra._napari._layers import (
        InteractionVector,
        LandscapeSurface,
        MoleculesLayer,
    )


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
                pass  # TODO: set to a gradient

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
        self.dim_label = QtWrappedLabel("view mode:")
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
        if isinstance(self._layer, MoleculesLayer):
            df = self._layer.molecules.features
        else:
            df = self._layer.net.features
        table = DataFrameView(value=df)

        napari.current_viewer().window.add_dock_widget(
            table, area="left", name=f"Features of {self._layer.name!r}"
        ).setFloating(True)

    def _copy_features(self):
        df: pd.DataFrame = self._layer.features
        df.to_clipboard(index=False)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QtW.QWidget]]:
        return [
            (self.feature_buttons_label, self.feature_btns),
        ]


class QtMoleculesControls(QtLayerControls):
    layer: MoleculesLayer
    MODE = PointsMode
    PAN_ZOOM_ACTION_NAME = "activate_points_pan_zoom_mode"
    TRANSFORM_ACTION_NAME = "activate_points_transform_mode"

    def __init__(self, layer) -> None:
        super().__init__(layer)
        self.panzoom_button.hide()
        self.transform_button.hide()

        self._on_editable_or_visible_change()

        # Setup widgets controls
        self._projection_mode_control = QtProjectionModeControl(self, layer)
        self._add_widget_controls(self._projection_mode_control)
        self._face_color_control = QtFaceControls(
            self,
            layer,
        )
        self._add_widget_controls(self._face_color_control)
        self._border_color_control = QtBorderControls(self, layer)
        self._add_widget_controls(self._border_color_control)
        self._text_visibility_control = QtTextVisibilityControl(self, layer)
        self._add_widget_controls(self._text_visibility_control)
        self._out_slice_checkbox_control = QtOutSliceCheckBoxControl(self, layer)
        self._add_widget_controls(self._out_slice_checkbox_control)
        self._point_state_control = QtPointStateControl(self, layer)
        self._add_widget_controls(self._point_state_control)
        self._has_features_control = QtHasFeaturesControls(self, layer)
        self._add_widget_controls(self._has_features_control)

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()


class QtLandscapeSurfaceControls(QtSurfaceControls):
    layer: LandscapeSurface

    def __init__(self, layer: LandscapeSurface) -> None:
        super().__init__(layer)
        self.panzoom_button.hide()
        self.transform_button.hide()
        layout: QFormLayout = self.layout()
        self.levelSlider = QLabeledDoubleSlider(orientation=Qt.Orientation.Horizontal)
        self.levelSlider.setRange(layer._level_min, layer._level_max)
        layout.addRow("level:", self.levelSlider)
        self.levelSlider.sliderReleased.connect(self._change_level)
        self.levelSlider.setSingleStep(0.001)
        self.levelSlider.setValue(layer.level)
        self.levelSlider.setToolTip("Threshold level for surface rendering")
        layer.events.level.connect(self._on_level_change)

        self.resolution = QtW.QDoubleSpinBox()
        layout.addRow("resolution:", self.resolution)
        self.resolution.setRange(0.1, 10)
        self.resolution.setSingleStep(0.05)
        self.resolution.setToolTip("Resolution of the surface")
        self.resolution.setValue(layer.resolution)
        self.resolution.valueChanged.connect(self._change_resolution)
        layer.events.resolution.connect(self._on_resolution_change)

        self.wireWidth = QtW.QDoubleSpinBox()
        self.wireWidth.setRange(0.0, 10.0)
        layout.addRow("wire width:", self.wireWidth)
        self.wireWidth.valueChanged.connect(self._change_wire_width)
        self.wireWidth.setSingleStep(0.05)
        self.wireWidth.setValue(layer.wireframe.width)
        self.wireWidth.setToolTip("Width of the wireframe")
        layer.wireframe.events.width.connect(self._on_wire_width_change)

        self.showMinCheckBox = QtW.QCheckBox("show min")
        layout.addRow(self.showMinCheckBox)
        self.showMinCheckBox.setChecked(layer.show_min)
        self.showMinCheckBox.stateChanged.connect(self._change_show_min)
        self.showMinCheckBox.setToolTip(
            "Show the surface of the minimum values of the energy"
        )
        layer.events.show_min.connect(self._on_show_min_change)

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
        self.layer.level = self.levelSlider.value()

    def _change_resolution(self):
        self.layer.resolution = self.resolution.value()

    def _change_show_min(self, state):
        self.layer.show_min = state == Qt.CheckState.Checked

    def _change_wire_width(self, value):
        self.layer.wireframe.width = value


class QtInteractionControls(QtLayerControls):
    layer: InteractionVector
    MODE = BaseMode
    PAN_ZOOM_ACTION_NAME = "activate_tracks_pan_zoom_mode"
    TRANSFORM_ACTION_NAME = "activate_tracks_transform_mode"

    def __init__(self, layer: InteractionVector) -> None:
        super().__init__(layer)
        self.panzoom_button.hide()
        self.transform_button.hide()
        # Setup widgets controls
        self._width_spinbox_control = QtWidthSpinBoxControl(self, layer)
        self._add_widget_controls(self._width_spinbox_control)
        self._projection_mode_control = QtProjectionModeControl(self, layer)
        self._add_widget_controls(self._projection_mode_control)
        self._vector_style_combobox_control = QtVectorStyleComboBoxControl(self, layer)
        self._add_widget_controls(self._vector_style_combobox_control)
        self._edge_color_feature_control = QtEdgeColorFeatureControl(self, layer)
        self._add_widget_controls(self._edge_color_feature_control)
        self._out_slice_checkbox_control = QtOutSliceCheckBoxControl(self, layer)
        self._add_widget_controls(self._out_slice_checkbox_control)

        self._has_features_control = QtHasFeaturesControls(self, layer)
        self._add_widget_controls(self._has_features_control)

        self.autoContrastBtn = QtW.QPushButton("Auto Contrast", self)
        self.layout().addWidget(self.autoContrastBtn)
        self.autoContrastBtn.clicked.connect(self._auto_contrast_edge)

    def _auto_contrast_edge(self):
        pname = self._edge_color_feature_control.color_feature_box.currentText()
        values = self.layer.features[pname]
        self.layer.edge_contrast_limits = values.min(), values.max()
        self.layer.refresh()


def _first_or(arr: np.ndarray, default):
    """Get the first element of the array or the default value."""
    if arr.size > 0:
        return arr[0]
    return default


def install_custom_layers():
    from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls

    from cylindra._napari._layers import (
        InteractionVector,
        LandscapeSurface,
        MoleculesLayer,
    )

    layer_to_controls[MoleculesLayer] = QtMoleculesControls
    layer_to_controls[LandscapeSurface] = QtLandscapeSurfaceControls
    layer_to_controls[InteractionVector] = QtInteractionControls
