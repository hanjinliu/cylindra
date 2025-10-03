from __future__ import annotations

from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.qt_surface_controls import QtSurfaceControls
from napari._qt.layer_controls.widgets import (
    QtOutSliceCheckBoxControl,
    QtProjectionModeControl,
    QtTextVisibilityControl,
)
from napari._qt.layer_controls.widgets._vectors import (
    QtEdgeColorFeatureControl,
    QtVectorStyleComboBoxControl,
    QtWidthSpinBoxControl,
)
from napari.layers.base._base_constants import Mode as BaseMode
from napari.layers.points._points_constants import Mode as PointsMode
from napari.utils.events import disconnect_events
from qtpy import QtWidgets as QtW

from cylindra._napari._sub_controls import (
    QtBorderControls,
    QtFaceControls,
    QtHasFeaturesControls,
    QtLandscapeSubControls,
    QtPointStateControl,
)

if TYPE_CHECKING:
    from cylindra._napari._layers import (
        InteractionVector,
        LandscapeSurface,
        MoleculesLayer,
    )


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
        self._landscape_sub_controls = QtLandscapeSubControls(self, layer)
        self._add_widget_controls(self._landscape_sub_controls)


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
