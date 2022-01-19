"""Make MTProps available in napari-plugin engine."""

from napari_plugin_engine import napari_hook_implementation
from .widget import MTPropsWidget

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return MTPropsWidget