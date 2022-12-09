from __future__ import annotations

from typing import TYPE_CHECKING
from magicclass import MagicTemplate, magicclass, field, nogui, vfield
from magicclass.ext.pyqtgraph import QtPlotCanvas
from cylindra.const import PropertyNames as H

if TYPE_CHECKING:
    from cylindra._list import DataFrameList

@magicclass
class DataFrameListWidget(MagicTemplate):
    def _get_choices(self, *_) -> list[str]:
        if not self._dfl:
            return []
        return self._dfl[0].columns.tolist()

    plt = field(QtPlotCanvas)
    data_index = vfield(int).with_options(max=0)
    column = vfield(str).with_choices(_get_choices)

    def __init__(self) -> None:
        self._dfl: DataFrameList = None

    def __post_init__(self) -> None:
        self.plt.background_color = [0.96, 0.96, 0.96, 1]
        self.plt.legend.visible = False
        self.plt.show_grid(x=True, y=True, alpha=0.5)
        self.plt.xlabel = "distance (nm)"
    
    @nogui
    def set_data(self, dfl: DataFrameList):
        self._dfl = dfl
        self["data_index"].max = len(dfl) - 1
        self.reset_choices()
    
    @data_index.connect
    @column.connect
    def _replot(self):
        index = self.data_index
        if col := self.column:
            self.plt.layers.clear()
            self.plt.add_curve(
                self._dfl[index][H.splDistance].values,
                self._dfl[index][col].values,
                color="blue",
                lw=2,
            )

    @column.connect
    def _auto_range(self):
        self.plt.auto_range()
