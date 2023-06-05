from __future__ import annotations
import polars as pl

from magicgui.widgets import SpinBox, Container, Label, FloatSpinBox, CheckBox
from magicgui.types import Undefined
from magicclass.widgets import ScrollableContainer


class ProtofilamentEdit(ScrollableContainer[Container[SpinBox]]):
    def __init__(
        self, value=Undefined, *, labels=True, nullable=False, **kwargs
    ) -> None:
        super().__init__(labels=labels, **kwargs)
        self.changed.disconnect()
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


class OffsetEdit(Container[FloatSpinBox]):
    def __init__(self, value=Undefined, *, nullable=False, **kwargs):
        self._offset_y = FloatSpinBox(
            value=0.0,
            label="axial (nm)",
            min=-100,
            max=100,
            step=0.1,
            tooltip="Offset in the axial direction.",
        )
        self._offset_a = FloatSpinBox(
            value=0.0,
            label="angular (deg)",
            min=-180,
            max=180,
            step=0.1,
            tooltip="Offset in the angular direction.",
        )
        super().__init__(
            widgets=[self._offset_y, self._offset_a], labels=True, **kwargs
        )
        self.changed.disconnect()
        self._offset_y.changed.connect(self._on_value_change)
        self._offset_a.changed.connect(self._on_value_change)
        if value is Undefined:
            value = (0.0, 0.0)
        self.value = value

    def _on_value_change(self):
        self.changed.emit(self.value)

    @property
    def value(self) -> tuple[float, float]:
        """Value of the widget"""
        return (self._offset_y.value, self._offset_a.value)

    @value.setter
    def value(self, val: tuple[float, float]) -> None:
        y, a = val
        with self.changed.blocked():
            self._offset_y.value = y
            self._offset_a.value = a
        self.changed.emit((y, a))


# class CheckBoxes(ScrollableContainer[CheckBox]):
#     def __init__(self, value=Undefined, *, labels=True, nullable=False, **kwargs) -> None:
#         super().__init__(labels=labels, **kwargs)
#         self.value = value
