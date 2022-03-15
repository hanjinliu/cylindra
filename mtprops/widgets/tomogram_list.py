import os.path
from typing import Any, List, Tuple

from magicclass import (
    magicclass,
    magictoolbar,
    field,
    MagicTemplate,
    )
from magicclass.types import Bound
from magicclass.utils import to_clipboard

from ..components import MtTomogram


@magicclass(widget_type="scrollable", labels=False)
class TomogramList(MagicTemplate):
    """List of tomograms that have been loaded to the widget."""        
    _tomogram_list: List[MtTomogram]
    
    def __init__(self):
        self._tomogram_list: List[MtTomogram] = []
        self._metadata_list: List[dict[str, Any]] = []
        
    def _get_tomograms(self, _=None) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        for i, tomo in enumerate(self._tomogram_list):
            try:
                parts = tomo.source.parts
                name = os.path.join(*parts[-2:])
            except Exception:
                name = f"Tomogram<{hex(id(tomo))}>"
            out.append((name, i))
        return out
    
    @magictoolbar
    class Tools(MagicTemplate):
        def Load(self): ...
        def Copy_path(self): ...
        def Delete(self): ...
    
    tomograms = field(int, widget_type="RadioButtons", options={"choices": _get_tomograms}, record=False)
    
    @Tools.wraps
    def Load(self, i: Bound[tomograms]):
        """Load selected tomogram into the viewer."""
        from .main import MTPropsWidget
        parent = self.find_ancestor(MTPropsWidget)
        tomo: MtTomogram = self._tomogram_list[i]
        if tomo is parent.tomogram:
            return None
        # Load dask again. Here, lowpass filter is already applied so that cutoff frequency
        # should be set to 0.
        parent._send_tomogram_to_viewer(tomo)
        parent._current_ft_size = tomo.metadata.get("ft_size", None)
        
        parent._init_layers()
        parent._init_widget_state()
        if tomo.splines:
            parent._update_splines_in_images()
            parent.Sample_subtomograms()
        else:
            parent.Panels.overview.layers.clear()
        
        return None
    
    @Tools.wraps
    def Copy_path(self, i: Bound[tomograms]):
        """Copy the path to the image file of the selected tomogram."""
        tomo: MtTomogram = self._tomogram_list[i]
        to_clipboard(str(tomo.source))
            
    @Tools.wraps
    def Delete(self, i: Bound[tomograms]):
        """Delete selected tomogram from the list."""
        from .main import MTPropsWidget
        parent = self.find_ancestor(MTPropsWidget)
        tomo: MtTomogram = self._tomogram_list[i]
        if tomo is parent.tomogram:
            raise ValueError("Tomogram is active now so cannot be deleted.")
        tomo = self._tomogram_list.pop(i)
        del tomo
        self.tomograms.reset_choices()
