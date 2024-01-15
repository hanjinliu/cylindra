from typing import TYPE_CHECKING

from magicclass import MagicTemplate, field, magicclass, vfield
from magicclass.ext.pyqtgraph import QtPlotCanvas
from napari.utils.colormaps import label_colormap

if TYPE_CHECKING:
    from acryo.classification import PcaClassifier


@magicclass(record=False)
class PcaViewer(MagicTemplate):
    def __init__(self, pca: "PcaClassifier"):
        self._pca = pca
        self._transform = None

    def __post_init__(self):
        self.reset_choices()
        self.choicex = 0
        self.choicey = 1

    @property
    def pca(self) -> "PcaClassifier":
        return self._pca

    @property
    def transform(self):
        if self._transform is None:
            self._transform = self.pca.get_transform()
        return self._transform

    def _get_choices(self, w=None) -> list[tuple[str, int]]:
        return [(f"PC-{i}", i) for i in range(self.pca.n_components)]

    choicex = vfield(int, label="X axis").with_choices(_get_choices)
    choicey = vfield(int, label="Y axis").with_choices(_get_choices)
    canvas = field(QtPlotCanvas)

    @choicex.connect
    @choicey.connect
    def _update_plot(self):
        if self.choicex is None or self.choicey is None:
            return
        pca = self.pca
        transformed = self.transform
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
        self.canvas.legend.visible = False
