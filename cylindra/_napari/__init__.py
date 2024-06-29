import warnings

from cylindra._napari._layers import LandscapeSurface, MoleculesLayer

__all__ = ["MoleculesLayer", "LandscapeSurface"]

try:
    # Here, a private files are used.
    from ._layer_controls import install_custom_layers

    install_custom_layers()
except ImportError:
    pass

del install_custom_layers

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="cylindra._napari",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="napari",
)
