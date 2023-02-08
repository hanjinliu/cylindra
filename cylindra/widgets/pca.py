from typing import TYPE_CHECKING
import numpy as np

from magicgui.widgets import FloatSlider
from magicclass import magicclass, field, vfield, MagicTemplate
from magicclass.ext.pyqtgraph import QtPlotCanvas
from magicclass.ext.vispy import Vispy3DCanvas

from napari.utils.colormaps import label_colormap

if TYPE_CHECKING:
    from acryo.classification import PcaClassifier

@magicclass(layout="horizontal", widget_type="split", record=False)
class PcaViewer(MagicTemplate):
    @magicclass
    class Plot(MagicTemplate):
        def _get_choices(self, w=None) -> list[tuple[str, int]]:
            try:
                pca = self.find_ancestor(PcaViewer, cache=True)._pca
            except RuntimeError:
                return []
            return [(f"PC-{i}", i) for i in range(pca.n_components)]

        choicex = vfield(int, label="X axis").with_choices(_get_choices)
        choicey = vfield(int, label="Y axis").with_choices(_get_choices)
        canvas = field(QtPlotCanvas)
        
        @choicex.connect
        @choicey.connect
        def _update_plot(self):
            if self.choicex is None or self.choicey is None:
                return
            pca = self.find_ancestor(PcaViewer, cache=True)._pca
            transformed = pca.get_transform()
            self.canvas.layers.clear()

            colors = label_colormap(pca.n_clusters + 1).colors[1:]
            for i in range(pca.n_clusters):
                sl = pca.labels == i
                color = colors[i]
                self.canvas.add_scatter(
                    transformed[sl, self.choicex],
                    transformed[sl, self.choicey],
                    size=3,
                    color=color,
                    name=f"Cluster-{i}",
                )
            self.canvas.xlabel = f"PC-{self.choicex}"
            self.canvas.ylabel = f"PC-{self.choicey}"
    
    @magicclass
    class BaseImage(MagicTemplate):
        def _get_choices(self, w=None) -> list[tuple[str, int]]:
            try:
                pca = self.find_ancestor(PcaViewer, cache=True)._pca
            except RuntimeError:
                return []
            return [(f"Base-{i}", i) for i in range(pca.n_components)]
            
        base_image = vfield(int).with_choices(_get_choices)
        iso_threshold = field(label="Iso threshold", widget_type=FloatSlider)
        canvas = field(Vispy3DCanvas)
        
        @base_image.connect
        def _update_canvas(self):
            pca = self.find_ancestor(PcaViewer, cache=True)._pca
            self._image.data = pca.get_bases()[self.base_image]
            self.iso_threshold.min = self._image.data.min()
            self.iso_threshold.max = self._image.data.max()
        
        @iso_threshold.connect
        def _update_threshold(self, val: float):
            self._image.iso_threshold = val
            
        def __post_init__(self):
            self._image = self.canvas.add_image(np.arange(8).reshape(2, 2, 2), rendering="iso")
            self.iso_threshold.min = 0
            self.iso_threshold.max = 7
            self.iso_threshold.value = 1.0

    def __init__(self, pca: "PcaClassifier"):
        self._pca = pca
    
    def __post_init__(self):
        self.reset_choices()
        self.Plot.choicex = 0
        self.Plot.choicey = 1
        self.BaseImage._update_canvas()