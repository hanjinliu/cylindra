from __future__ import annotations

import polars as pl
from magicclass import MagicTemplate, magicclass, field, vfield
from magicclass.ext.pyqtgraph import QtPlotCanvas
from cylindra.const import PropertyNames as H, MoleculesHeader as Mole, IDName


@magicclass
class LocalPropsViewer(MagicTemplate):
    plt = field(QtPlotCanvas)

    def _get_data_index(self, w=None) -> list[str]:
        return [(f"image={k[0]}, spline={k[1]}", k) for k in self._groups.keys()]

    def _get_columns(self, w=None) -> list[str]:
        return [H.yPitch, H.skewAngle, H.nPF, H.riseAngle]

    data_index = field(str, label="Data").with_choices(_get_data_index)
    column = vfield(str, label="Column name").with_choices(_get_columns)

    def __init__(self) -> None:
        self._groups = {}

    def _set_localprops(self, df: pl.DataFrame) -> None:
        self._groups = dict(
            df.groupby(by=[Mole.image, IDName.spline], maintain_order=True)
        )
        self.reset_choices()

    def __post_init__(self) -> None:
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
