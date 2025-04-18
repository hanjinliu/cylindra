from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from fnmatch import fnmatch
from typing import TYPE_CHECKING

import napari
import numpy as np
from magicclass.ext.polars import DataFrameView
from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari._qt.layer_controls.qt_surface_controls import QtSurfaceControls
from napari._qt.layer_controls.qt_vectors_controls import QtVectorsControls
from qtpy import QtCore
from qtpy import QtWidgets as QtW
from qtpy.QtCore import Qt
from superqt import QEnumComboBox, QLabeledDoubleSlider

if TYPE_CHECKING:
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


class QtMoleculesControls(QtPointsControls):
    layer: MoleculesLayer

    def __init__(self, layer: MoleculesLayer) -> None:
        super().__init__(layer)
        # hide buttons to avoid mutable operations
        self.panzoom_button.hide()
        self.select_button.hide()
        self.addition_button.hide()
        self.delete_button.hide()
        self.transform_button.hide()

        self.faceColorEdit.color_changed.disconnect()
        self.faceColorEdit.color_changed.connect(self._change_face_color)
        self.borderColorEdit.color_changed.disconnect()
        self.borderColorEdit.setColor(_first_or(layer.border_color, "dimgray"))
        self.borderColorEdit.color_changed.connect(self._change_border_color)

        slider = QLabeledDoubleSlider(orientation=Qt.Orientation.Horizontal)
        slider.setRange(0.1, 12.0)
        slider.setSingleStep(0.02)
        slider.setValue(layer.point_size)
        slider.setToolTip("Point size of all the molecules")
        slider.valueChanged.connect(self._change_point_size)
        self.pointSizeSlider = slider

        self.borderWidth = QtW.QDoubleSpinBox()
        self.borderWidth.setRange(0.0, 1.0)
        self.borderWidth.valueChanged.connect(self._change_border_width)
        self.borderWidth.setSingleStep(0.05)
        self.borderWidth.setValue(_first_or(layer.border_width, 0.05))
        self.borderWidth.setToolTip("Width of the molecule borders")
        layer.events.border_width.connect(self._on_border_width_change)

        self.dimComboBox = QEnumComboBox(enum_class=ViewDimension)
        self.dimComboBox.currentEnumChanged.connect(self._change_dim)
        self.dimComboBox.setCurrentEnum(ViewDimension(layer.view_ndim))

        layout: QFormLayout = self.layout()
        layout.addRow("point size:", self.pointSizeSlider)
        layout.addRow("border width:", self.borderWidth)
        layout.addRow("view mode:", self.dimComboBox)

        btns = QtW.QHBoxLayout()
        self.showFeatureButton = QtW.QPushButton("show", self)
        self.showFeatureButton.setToolTip("Show features of the molecules in a table")
        self.showFeatureButton.clicked.connect(self._show_features)
        btns.addWidget(self.showFeatureButton)
        self.copyFeatureButton = QtW.QPushButton("copy", self)
        self.copyFeatureButton.setToolTip("Copy features of the molecules to clipboard")
        self.copyFeatureButton.clicked.connect(self._copy_features)
        btns.addWidget(self.copyFeatureButton)
        layout.addRow("features:", btns)

        self.propertyFilter = QtW.QLineEdit()
        self.propertyFilter.setPlaceholderText("Filter status tip ...")
        self.propertyFilter.setToolTip(
            "Filter the mouse-hover status tip by the column name.\n"
            "e.g. 'align-* will match all columns starting with 'align-'\n"
        )
        self.propertyFilter.editingFinished.connect(self._set_property_filter)
        layout.addRow("filter status:", self.propertyFilter)

        layout.removeRow(self.sizeSlider)
        layout.removeRow(self.symbolComboBox)
        layer.events.point_size.connect(self._on_point_size_change)
        layer.events.view_ndim.connect(self._on_dim_change)

    def _on_point_size_change(self, event):
        with qt_signals_blocked(self.pointSizeSlider):
            self.pointSizeSlider.setValue(event.value)

    def _on_dim_change(self, event):
        with qt_signals_blocked(self.dimComboBox):
            self.dimComboBox.setCurrentEnum(ViewDimension(event.value))

    def _on_border_width_change(self, event):
        with qt_signals_blocked(self.borderWidth):
            self.borderWidth.setValue(_first_or(self.layer.border_width, 0.05))

    def _change_point_size(self, value: float):
        self.layer.point_size = value

    def _change_border_width(self, value: float):
        self.layer.border_width = value

    def _change_dim(self, value: ViewDimension):
        self.layer.view_ndim = value.value

    def _change_face_color(self, value):
        self.layer.face_color = value

    def _change_border_color(self, value):
        self.layer.border_color = value

    def _on_current_face_color_change(self):
        pass

    def _show_features(self):
        df = self.layer.molecules.features
        table = DataFrameView(value=df)

        napari.current_viewer().window.add_dock_widget(
            table, area="left", name=f"Features of {self.layer.name!r}"
        ).setFloating(True)

    def _copy_features(self):
        df = self.layer.features
        df.to_clipboard(index=False)

    def _set_property_filter(self):
        text = self.propertyFilter.text()
        if text:
            self.layer._property_filter = lambda x: fnmatch(x, text)
        else:
            self.layer._property_filter = lambda _: True


class QtLandscapeSurfaceControls(QtSurfaceControls):
    layer: LandscapeSurface

    def __init__(self, layer: LandscapeSurface) -> None:
        super().__init__(layer)
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


class QtInteractionControls(QtVectorsControls):
    layer: InteractionVector

    def __init__(self, layer: InteractionVector) -> None:
        super().__init__(layer)
        layout: QFormLayout = self.layout()

        btns = QtW.QHBoxLayout()
        self.showFeatureButton = QtW.QPushButton("show", self)
        self.showFeatureButton.setToolTip("Show features of the interaction in a table")
        self.showFeatureButton.clicked.connect(self._show_features)
        btns.addWidget(self.showFeatureButton)
        self.copyFeatureButton = QtW.QPushButton("copy", self)
        self.copyFeatureButton.setToolTip(
            "Copy features of the interaction to clipboard"
        )
        self.copyFeatureButton.clicked.connect(self._copy_features)
        btns.addWidget(self.copyFeatureButton)
        layout.addRow("features:", btns)

        layout.removeRow(self.lengthSpinBox)
        layout.removeRow(self.color_mode_comboBox)
        self.autoContrastBtn = QtW.QPushButton("Auto Contrast", self)
        layout.addWidget(self.autoContrastBtn)
        self.autoContrastBtn.clicked.connect(self._auto_contrast_edge)

    def _show_features(self):
        df = self.layer.net.features
        table = DataFrameView(value=df)

        napari.current_viewer().window.add_dock_widget(
            table, area="left", name=f"Features of {self.layer.name!r}"
        ).setFloating(True)

    def _copy_features(self):
        df = self.layer.features
        df.to_clipboard(index=False)

    def _auto_contrast_edge(self):
        pname = self.color_prop_box.currentText()
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
