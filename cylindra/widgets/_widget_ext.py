from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Callable, Iterable
import random
from magicgui.widgets.bases._value_widget import ValueWidget

from qtpy import QtWidgets as QtW, QtCore, QtGui
from qtpy.QtCore import Qt

from magicgui.widgets import (
    SpinBox,
    Container,
    Label,
    FloatSpinBox,
    LineEdit,
    PushButton,
    protocols,
)
from magicgui.widgets.bases import CategoricalWidget
from magicgui.types import ChoicesType, Undefined
from magicgui.backends._qtpy import widgets as backend_qtw
from magicclass.widgets import ScrollableContainer


class ProtofilamentEdit(ScrollableContainer[Container[SpinBox]]):
    """Widget for editing protofilament prepend/append numbers."""

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
    """Widget for editing offset in the axial and angular direction."""

    def __init__(self, value=Undefined, *, nullable=False, **kwargs):
        # fmt: off
        self._offset_y = FloatSpinBox(value=0.0, label="axial (nm)", min=-100, max=100, step=0.1, tooltip="Offset in the axial direction.")
        self._offset_a = FloatSpinBox(value=0.0, label="angular (deg)", min=-180, max=180, step=0.1, tooltip="Offset in the angular direction.")
        super().__init__(widgets=[self._offset_y, self._offset_a], labels=True, **kwargs)
        # fmt: on
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


def _rotation_widgets(max: float = 180.0):
    _max = FloatSpinBox(
        label="max",
        max=max,
        step=0.1,
        tooltip="Maximum rotation angle in external degree",
    )
    _step = FloatSpinBox(
        label="step",
        max=max,
        step=0.1,
        tooltip="Step of rotation angle in degree (rotation angles will be calculated by adding/subtracting this value, with 0.0 as the center)",
    )
    return _max, _step


class RotationEdit(Container[Container[FloatSpinBox]]):
    """Widget for editing Euler rotation angles for subtomogram averaging."""

    def __init__(self, value=Undefined, *, nullable=False, **kwargs):
        # fmt: off
        self._z_max, self._z_step = _rotation_widgets(180)
        self._y_max, self._y_step = _rotation_widgets(180)
        self._x_max, self._x_step = _rotation_widgets(90)
        _z_container = Container(widgets=[self._z_max, self._z_step], labels=True, label="Z", layout="horizontal")
        _z_container.margins = (0, 0, 0, 0)
        _y_container = Container(widgets=[self._y_max, self._y_step], labels=True, label="Y", layout="horizontal")
        _y_container.margins = (0, 0, 0, 0)
        _x_container = Container(widgets=[self._x_max, self._x_step], labels=True, label="X", layout="horizontal")
        _x_container.margins = (0, 0, 0, 0)
        super().__init__(widgets=[_z_container, _y_container, _x_container], labels=True, **kwargs)
        # fmt: on
        self.changed.disconnect()
        self._z_max.changed.connect(self._on_value_change)
        self._z_step.changed.connect(self._on_value_change)
        self._y_max.changed.connect(self._on_value_change)
        self._y_step.changed.connect(self._on_value_change)
        self._x_max.changed.connect(self._on_value_change)
        self._x_step.changed.connect(self._on_value_change)
        if value is Undefined:
            value = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        self.value = value

    def _on_value_change(self):
        self.changed.emit(self.value)

    @property
    def value(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Value of the widget"""
        return (
            (self._z_max.value, self._z_step.value),
            (self._y_max.value, self._y_step.value),
            (self._x_max.value, self._x_step.value),
        )

    @value.setter
    def value(self, val: tuple[float, float]) -> None:
        (z_max, z_step), (y_max, y_step), (x_max, x_step) = val
        with self.changed.blocked():
            self._z_max.value = z_max
            self._z_step.value = z_step
            self._y_max.value = y_max
            self._y_step.value = y_step
            self._x_max.value = x_max
            self._x_step.value = x_step
        self.changed.emit(val)


class RandomSeedEdit(Container):
    """Widget for editing random seed values."""

    def __init__(self, value=Undefined, *, nullable=False, **kwargs: Any) -> None:
        self._seeds = LineEdit(
            value="0, 1, 2, 3, 4",
            tooltip=(
                "Random seed values. You can use any Python literals in this box.\n"
                "e.g.) `0, 1, 2, 3, 4`, `range(3)`"
            ),
        )
        self._btn = PushButton(
            label="Add", tooltip="Add a new random seed value to the list."
        )
        self._btn.max_width = 32
        super().__init__(
            widgets=[self._seeds, self._btn],
            layout="horizontal",
            labels=False,
            **kwargs,
        )
        self.changed.disconnect()
        # NOTE: don't relay value-change event of the line edit to the parents. Evaluation usually
        # fails and it is potentially dangerous or time consuming.
        self._btn.clicked.connect(self._generate_new_seed)
        self.value = value

    def _on_value_change(self):
        self.changed.emit(self.value)

    def _generate_new_seed(self):
        try:
            val = self.value
        except Exception:
            val = []
        count = 0
        while (newval := random.randint(0, 1e9 - 1)) in val:
            # sample an identical value.
            count += 1
            if count > 1000:  # just in case!
                raise RecursionError("Failed to generate a new random seed value.")
        val.append(newval)
        self.value = val

    @property
    def value(self) -> list[int]:
        _str = self._seeds.value
        if _str == "":
            raise ValueError("Empty string is not allowed.")
        out = eval(_str, {"__builtins__": {"range": range}}, {})
        if isinstance(out, int):
            return [out]
        for i, x in enumerate(out):
            if not isinstance(x, int):
                raise TypeError(f"Invalid type at {i}-th component: {type(x)}")
        return list(out)

    @value.setter
    def value(self, val: Any):
        if not isinstance(val, str):
            if hasattr(val, "__iter__"):
                val = ", ".join(str(int(v)) for v in val)
            else:
                raise TypeError(f"Invalid type for value: {type(val)}")
        self._seeds.value = val


@contextmanager
def _signals_blocked(obj: QtW.QWidget):
    before = obj.blockSignals(True)
    try:
        yield
    finally:
        obj.blockSignals(before)


class _ListWidget(QtW.QListWidget):
    def __init__(self, parent: QtW.QWidget | None = None):
        super().__init__(parent)
        self.setSelectionMode(QtW.QAbstractItemView.SelectionMode.NoSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.itemEntered.connect(self._item_entered)
        self._mouse_down = False

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        self._mouse_down = True
        return super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        self._mouse_down = False
        return super().mouseReleaseEvent(e)

    def _item_entered(self, item: QtW.QListWidgetItem):
        if self._mouse_down:
            self.itemClicked.emit(item)

    def sizeHint(self) -> QtCore.QSize:
        size = super().sizeHint()
        size.setHeight(int(size.height() / 2))
        return size


class BaseSelect(backend_qtw.QBaseValueWidget, protocols.CategoricalWidgetProtocol):
    _qwidget: _ListWidget

    def __init__(self, **kwargs) -> None:
        super().__init__(_ListWidget, "isChecked", "setCurrentIndex", "", **kwargs)
        self._qwidget.itemChanged.connect(self._emit_data)
        self._qwidget.itemClicked.connect(self._toggle_item_checked)

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
