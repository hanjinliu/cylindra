from __future__ import annotations

from magicclass import MagicTemplate, magicclass, field, vfield
from magicclass.ext.pyqtgraph import QtPlotCanvas
from cylindra.const import PropertyNames as H, MoleculesHeader as Mole, IDName
from cylindra.project import ProjectSequence

@magicclass
class LocalPropsViewer(MagicTemplate):
    plt = field(QtPlotCanvas)
    data_index = field(str).with_choices([])
    column = vfield(str).with_choices([H.yPitch, H.skewAngle, H.nPF, H.riseAngle])

    def __init__(self, seq: ProjectSequence) -> None:
        if not isinstance(seq, ProjectSequence):
            raise TypeError(f"Expected ProjectSequence, got {type(seq)}.")
        self._df = seq.localprops()
        self._groups = dict(self._df.groupby(by=[Mole.image, IDName.spline]))

    def __post_init__(self) -> None:
        self.plt.background_color = [0.96, 0.96, 0.96, 1]
        self.plt.legend.visible = False
        self.plt.show_grid(x=True, y=True, alpha=0.5)
        self.plt.xlabel = "distance (nm)"
        self.data_index.choices = list(map(str, self._groups.keys()))
        self.reset_choices()
    
    
    @data_index.connect
    @column.connect
    def _replot(self):
        key = self.data_index.value
        df = self._groups[key]
        if col := self.column:
            self.plt.layers.clear()
            self.plt.add_curve(
                df[H.splDistance].to_numpy(),
                df[col].to_numpy(),
                color="blue",
                lw=2,
            )

    @column.connect
    def _auto_range(self):
        self.plt.auto_range()
