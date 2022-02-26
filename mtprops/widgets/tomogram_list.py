import os.path
from typing import List, Tuple

from magicclass import (
    magicclass,
    magictoolbar,
    field,
    Bound,
    MagicTemplate,
    )

from magicclass.utils import to_clipboard

from ..components import MtTomogram


@magicclass(widget_type="scrollable", labels=False)
class TomogramList(MagicTemplate):
    """List of tomograms that have been loaded to the widget."""        
    _tomogram_list: List[MtTomogram]
    
    def __init__(self):
        self._tomogram_list = []
        
    def _get_tomograms(self, widget=None) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        for i, tomo in enumerate(self._tomogram_list):
            try:
                d, name0 = os.path.split(tomo.metadata["source"])
                _, name1 = os.path.split(d)
                name = os.path.join(name1, name0)
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
        parent.tomogram = tomo
        
        # Load dask again. Here, lowpass filter is already applied so that cutoff frequency
        # should be set to 0.
        worker = parent._get_process_image_worker(
            tomo.image, 
            path=tomo.metadata["source"],
            binsize=tomo.metadata["binsize"], 
            light_bg=tomo.light_background, 
            cutoff=tomo.metadata["cutoff"],
            length=tomo.subtomo_length,
            width=tomo.subtomo_width,
            new=False
            )
        parent._last_ft_size = tomo.metadata.get("ft_size", None)
        parent._connect_worker(worker)
        worker.start()
        
        if tomo.splines:
            worker.finished.connect(parent._init_figures)
            worker.finished.connect(parent.Sample_subtomograms)
        else:
            worker.finished.connect(parent._init_layers)
            worker.finished.connect(parent._init_widget_params)
            worker.finished.connect(parent.Panels.overview.layers.clear)
        
        return None
    
    @Tools.wraps
    def Copy_path(self, i: Bound[tomograms]):
        """Copy the path to the image file of the selected tomogram."""
        tomo: MtTomogram = self._tomogram_list[i]
        if "source" in tomo.metadata:
            to_clipboard(tomo.metadata["source"])
            
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
