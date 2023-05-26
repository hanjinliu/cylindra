from __future__ import annotations
import polars as pl

from magicgui.widgets import SpinBox, Container, Label, CheckBox
from magicgui.types import Undefined
from magicclass.widgets import ScrollableContainer


class ProtofilamentEdit(ScrollableContainer[Container[SpinBox]]):
    def __init__(
        self, value=Undefined, *, labels=True, nullable=False, **kwargs
    ) -> None:
        super().__init__(labels=labels, **kwargs)
        self.value = value

    def _add_row(self, label: int, value: tuple[int, int]):
        val0, val1 = value
        row = Container(
            widgets=[
                SpinBox(
                    value=val0, min=-1000, tooltip="Number of molecules to prepend"
                ),
                SpinBox(value=val1, min=-1000, tooltip="Number of molecules to append"),
            ],
            layout="horizontal",
            label=str(label),
        )
        row.margins = (0, 0, 0, 0)
        self.append(row)
        row.changed.disconnect()
        row.changed.connect(self._on_changed)

    @property
    def value(self) -> dict[int, tuple[int, int]]:
        """Dict value of prepend/append numbers."""
        out: dict[int, tuple[int, int]] = {}
        for row in self:
            if not isinstance(row, Container):
                continue
            pf_id = int(row.label)
            vals = row[0].value, row[1].value
            out[pf_id] = vals
        return out

    @value.setter
    def value(self, val: dict[int, tuple[int, int]]):
        if val is Undefined:
            val = {}
        with self.changed.blocked():
            self.clear()
            for k, v in val.items():
                self._add_row(k, v)
        self.changed.emit(val)
        if len(val) == 0:
            self.append(Label(value="No protofilament info found."))

    def _on_changed(self):
        self.changed.emit(self.value)


# class CheckBoxes(ScrollableContainer[CheckBox]):
#     def __init__(self, value=Undefined, *, labels=True, nullable=False, **kwargs) -> None:
#         super().__init__(labels=labels, **kwargs)
#         self.value = value
