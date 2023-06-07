from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Callable, Iterable
from magicgui.widgets.bases._value_widget import ValueWidget

from qtpy import QtWidgets as QtW
from qtpy.QtCore import Qt

from magicgui.widgets import SpinBox, Container, Label, FloatSpinBox, protocols
from magicgui.widgets.bases import CategoricalWidget
from magicgui.types import ChoicesType, Undefined
from magicgui.backends._qtpy import widgets as backend_qtw
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


@contextmanager
def _signals_blocked(obj: QtW.QWidget):
    before = obj.blockSignals(True)
    try:
        yield
    finally:
        obj.blockSignals(before)


class BaseSelect(backend_qtw.QBaseValueWidget, protocols.CategoricalWidgetProtocol):
    _qwidget: QtW.QListWidget

    def __init__(self, **kwargs) -> None:
        super().__init__(QtW.QListWidget, "isChecked", "setCurrentIndex", "", **kwargs)
        self._qwidget.setSelectionMode(QtW.QAbstractItemView.SelectionMode.NoSelection)
        self._qwidget.itemChanged.connect(self._emit_data)
        self._qwidget.itemClicked.connect(self._toggle_item_checked)
        self._qwidget.itemEntered.connect(self._toggle_item_checked)
        self._qwidget.setMaximumHeight(80)
        self._qwidget.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

    def _emit_data(self):
        self._event_filter.valueChanged.emit(
            [d.data(Qt.ItemDataRole.UserRole) for d in self._iter_checked()]
        )

    def _toggle_item_checked(self, item: QtW.QListWidgetItem):
        if item.checkState() == Qt.CheckState.Checked:
            state = Qt.CheckState.Unchecked
        else:
            state = Qt.CheckState.Checked
        item.setCheckState(state)

    def _iter_checked(self):
        for i in range(self._qwidget.count()):
            item = self._qwidget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                yield item

    def _mgui_bind_change_callback(self, callback):
        self._event_filter.valueChanged.connect(callback)

    def _mgui_get_count(self) -> int:
        """Return the number of items in the dropdown."""
        return self._qwidget.count()

    def _mgui_get_choice(self, choice_name: str) -> list[Any]:
        items = self._qwidget.findItems(choice_name, Qt.MatchFlag.MatchExactly)
        return [i.data(Qt.ItemDataRole.UserRole) for i in items]

    def _mgui_get_current_choice(self) -> list[str]:  # type: ignore[override]
        return [i.text() for i in self._iter_checked()]

    def _mgui_get_value(self) -> Any:
        return [i.data(Qt.ItemDataRole.UserRole) for i in self._iter_checked()]

    def _mgui_set_value(self, value) -> None:
        if not isinstance(value, (list, tuple)):
            value = [value]
        selected_prev = self._iter_checked()
        with _signals_blocked(self._qwidget):
            for i in range(self._qwidget.count()):
                item = self._qwidget.item(i)
                if item.data(Qt.ItemDataRole.UserRole) in value:
                    item.setCheckState(Qt.CheckState.Checked)
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)
        selected_post = self._iter_checked()
        if selected_prev != selected_post:
            self._emit_data()

    def _mgui_set_choice(self, choice_name: str, data: Any) -> None:
        """Set data for ``choice_name``."""
        items = self._qwidget.findItems(choice_name, Qt.MatchFlag.MatchExactly)
        # if it's not in the list, add a new item
        if not items:
            item = QtW.QListWidgetItem(choice_name)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, data)
            self._qwidget.addItem(item)
        # otherwise update its data
        else:
            for item in items:
                item.setData(Qt.ItemDataRole.UserRole, data)

    def _mgui_set_choices(self, choices: Iterable[tuple[str, Any]]) -> None:
        """Set current items in categorical type ``widget`` to ``choices``."""
        choices_ = list(choices)
        if not choices_:
            self._qwidget.clear()
            return

        with _signals_blocked(self._qwidget):
            choice_names = [x[0] for x in choices_]
            selected_prev = self._iter_checked()
            # remove choices that no longer exist
            for i in reversed(range(self._qwidget.count())):
                if self._qwidget.item(i).text() not in choice_names:
                    self._qwidget.takeItem(i)
            # update choices
            for name, data in choices_:
                self._mgui_set_choice(name, data)
            selected_post = self._iter_checked()
        if selected_prev != selected_post:
            self._emit_data()

    def _mgui_del_choice(self, choice_name: str) -> None:
        """Delete choice_name."""
        for i in reversed(range(self._qwidget.count())):
            if self._qwidget.item(i).text() == choice_name:
                self._qwidget.takeItem(i)

    def _mgui_get_choices(self) -> tuple[tuple[str, Any], ...]:
        """Get available choices."""
        return tuple(
            (
                self._qwidget.item(i).text(),
                self._qwidget.item(i).data(Qt.ItemDataRole.UserRole),
            )
            for i in range(self._qwidget.count())
        )


class CheckBoxes(CategoricalWidget):
    _allow_multiple = True

    def __init__(
        self,
        value: Any = Undefined,
        choices: ChoicesType = (),
        *,
        allow_multiple: bool | None = None,
        bind: Any | Callable[[ValueWidget], Any] = Undefined,
        nullable: bool = False,
        **base_widget_kwargs: Any,
    ) -> None:
        base_widget_kwargs["widget_type"] = BaseSelect
        super().__init__(
            value,
            choices,
            allow_multiple=allow_multiple,
            bind=bind,
            nullable=nullable,
            **base_widget_kwargs,
        )

    def reset_choices(self, *_: Any) -> None:
        super().reset_choices(*_)
        self.value = self.choices
