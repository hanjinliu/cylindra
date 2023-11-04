from __future__ import annotations

from typing import TYPE_CHECKING
from enum import Enum
from contextlib import contextmanager
from qtpy import QtCore, QtWidgets as QtW
from qtpy.QtCore import Qt
from superqt import QEnumComboBox, QLabeledDoubleSlider
from napari._qt.layer_controls.qt_surface_controls import QtSurfaceControls
from napari._qt.layer_controls.qt_points_controls import QtPointsControls

if TYPE_CHECKING:
    from qtpy.QtWidgets import QFormLayout
    from ._layers import MoleculesLayer, LandscapeSurface


@contextmanager
def qt_signals_blocked(obj: QtCore.QObject):
    previous = obj.blockSignals(True)
    try:
        yield
    finally:
        obj.blockSignals(previous)


class ViewDimension(Enum):
    flat = 2
    sphere = 3


class QtMoleculesControls(QtPointsControls):
    layer: MoleculesLayer

    def __init__(self, layer: MoleculesLayer) -> None:
        super().__init__(layer)
        # hide unnecessary buttons to avoid mutable operations
        self.addition_button.hide()
        self.delete_button.hide()

        self.faceColorEdit.color_changed.disconnect()
        self.faceColorEdit.color_changed.connect(self._change_face_color)

        slider = QLabeledDoubleSlider(orientation=Qt.Orientation.Horizontal)
        slider.setRange(0.1, 12)
        slider.setSingleStep(0.02)
        slider.setValue(layer.point_size)
        slider.setToolTip("Point size of all the molecules")
        slider.valueChanged.connect(self._change_point_size)
        self.pointSizeSlider = slider

        self.dimComboBox = QEnumComboBox(enum_class=ViewDimension)
        self.dimComboBox.currentEnumChanged.connect(self._change_dim)
        self.dimComboBox.setCurrentEnum(ViewDimension(layer.view_ndim))

        layout: QFormLayout = self.layout()
        layout.addRow("point size:", self.pointSizeSlider)
        layout.addRow("view mode:", self.dimComboBox)
        layout.removeRow(self.sizeSlider)
        layout.removeRow(self.symbolComboBox)
        layout.removeRow(self.edgeColorEdit)
        layer.events.point_size.connect(self._on_point_size_change)
        layer.events.view_ndim.connect(self._on_dim_change)

    def _on_point_size_change(self, event):
        with qt_signals_blocked(self.pointSizeSlider):
            self.pointSizeSlider.setValue(event.value)

    def _change_point_size(self, value: float):
        self.layer.point_size = value

    def _on_dim_change(self, event):
        with qt_signals_blocked(self.dimComboBox):
            self.dimComboBox.setCurrentEnum(ViewDimension(event.value))

    def _on_face_color_change(self, value):
        with qt_signals_blocked(self.faceColorEdit):
            self.layer.face_color = value
            self.layer._update_thumbnail()

    def _change_dim(self, value: ViewDimension):
        self.layer.view_ndim = value.value

    def _change_face_color(self, value):
        self.layer.face_color = value

    def _on_current_face_color_change(self):
        pass


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
            if event.value:
                state = Qt.CheckState.Checked
            else:
                state = Qt.CheckState.Unchecked
            self.showMinCheckBox.setChecked(state)

    def _change_level(self):
        self.layer.level = self.levelSlider.value()

    def _change_resolution(self):
        self.layer.resolution = self.resolution.value()

    def _change_show_min(self, state):
        self.layer.show_min = state == Qt.CheckState.Checked


def install_custom_layers():
    from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls
    from ._layers import MoleculesLayer, LandscapeSurface

    layer_to_controls[MoleculesLayer] = QtMoleculesControls
    layer_to_controls[LandscapeSurface] = QtLandscapeSurfaceControls
