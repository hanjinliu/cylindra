from __future__ import annotations

import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from magicclass.widgets import ScrollableContainer
from magicgui.backends._qtpy import widgets as backend_qtw
from magicgui.types import ChoicesType, Undefined
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSpinBox,
    Label,
    LineEdit,
    PushButton,
    SpinBox,
    protocols,
    show_file_dialog,
)
from magicgui.widgets.bases import CategoricalWidget, ValueWidget
from qtpy import QtCore, QtGui
from qtpy import QtWidgets as QtW
from qtpy.QtCore import Qt


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
        self._offset_a = FloatSpinBox(value=0.0, label="angular (rad)", min=-180, max=180, step=0.1, tooltip="Offset in the angular direction.")
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


class RotationsEdit(Container[Container[FloatSpinBox]]):
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


class ParametrizedEnumEdit(Container):
    def __init__(self, value=Undefined, *, nullable=False, **kwargs):
        self._choice = self._get_choice_widget()
        self._value = self._get_value_widget()
        super().__init__(
            widgets=[self._choice, self._value],
            labels=False,
            layout="horizontal",
            **kwargs,
        )
        self.changed.disconnect()
        self._choice.changed.connect(self._on_value_change)
        self._value.changed.connect(self._on_value_change)
        if value is not Undefined:
            self.value = value

    def _get_value_widget(self) -> ValueWidget[float]:
        raise NotImplementedError()

    def _get_choice_widget(self) -> ValueWidget[str]:
        raise NotImplementedError()

    def _on_value_change(self):
        self.changed.emit(self.value)

    @property
    def value(self) -> tuple[str, float]:
        return self._choice.value, self._value.value

    @value.setter
    def value(self, val):
        self._choice.value, self._value.value = val


class SingleRotationEdit(ParametrizedEnumEdit):
    def _get_value_widget(self):
        return FloatSpinBox(
            value=0.0,
            min=-180,
            max=180,
            step=0.1,
            tooltip="Rotation angle in degree",
            name="degree",
        )

    def _get_choice_widget(self):
        return ComboBox(
            value="z", choices=["z", "y", "x"], tooltip="Rotation axis", name="axis"
        )


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
        self.margins = (0, 0, 0, 0)
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
        while (newval := random.randint(0, 99999999)) in val:
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
            return []
        out = eval(_str, {"__builtins__": {"range": range}}, {})
        if isinstance(out, int):
            return [out]
        for i, x in enumerate(out):
            if not isinstance(x, int):
                raise TypeError(f"Invalid type at {i}-th component: {type(x)}")
        return list(out)

    @value.setter
    def value(self, val: Any):
        if val == Undefined:
            val = "0"
        if not isinstance(val, str):
            if hasattr(val, "__iter__"):
                val = ", ".join(str(int(v)) for v in val)
            else:
                raise TypeError(f"Invalid type for value: {type(val)}")
        self._seeds.value = val


class MultiFileEdit(Container):
    def __init__(
        self,
        value=Undefined,
        *,
        filter: str | None = None,
        nullable=False,
        **kwargs: Any,
    ) -> None:
        # fmt: off
        self._add_btn = PushButton(label="Add", tooltip="Add new file paths.")
        self._toggle_btn = PushButton(label="Select", tooltip="Toggle file path deletion.")
        self._del_btn = PushButton(label="Delete", tooltip="Delete selected paths.", visible=False)
        self._add_btn.max_width = 60
        self._toggle_btn.max_width = 50
        self._del_btn.max_width = 50
        _cnt = Container(widgets=[self._del_btn, self._toggle_btn, self._add_btn], layout="horizontal")
        _cnt.margins = (0, 0, 0, 0)
        self._paths = ScrollableContainer[Container[LineEdit | CheckBox]](
            widgets=[], layout="vertical", labels=True
        )
        self._filter = filter
        super().__init__(widgets=[self._paths, _cnt], layout="vertical", labels=False, **kwargs)
        # fmt: on
        self.margins = (0, 0, 0, 0)
        self.changed.disconnect()
        self._add_btn.clicked.connect(self._add_files_from_dialog)
        self._toggle_btn.clicked.connect(self._toggle_delete_checkboxes)
        self._del_btn.clicked.connect(self._delete_checked_paths)
        self.value = value

    def _append_paths(self, paths: list[str]):
        for i, path in enumerate(paths):
            line = LineEdit(value=str(path))
            cbox = CheckBox(value=False, visible=False)
            _cnt = Container(
                widgets=[cbox, line], layout="horizontal", labels=False, label=f"({i}):"
            )
            _cnt.margins = (0, 0, 0, 0)
            self._paths.append(_cnt)
        self._toggle_btn.enabled = len(self._paths) > 0

    def _toggle_delete_checkboxes(self):
        visible = not self._del_btn.visible
        for wdt in self._paths:
            wdt[0].visible = visible
            if visible:
                wdt[0].value = False  # uncheck
        if visible:
            self._toggle_btn.text = "Cancel"
        else:
            self._toggle_btn.text = "Select"
        self._del_btn.visible = visible

    def _delete_checked_paths(self):
        to_delete = list[int]()
        for i, wdt in enumerate(self._paths):
            if wdt[0].value:
                to_delete.append(i)
        for i in sorted(to_delete, reverse=True):
            del self._paths[i]
        # relabel
        for i, wdt in enumerate(self._paths):
            wdt.label = f"({i}):"
        self._toggle_delete_checkboxes()

    @property
    def value(self) -> list[Path]:
        return [Path(wdt[1].value) for wdt in self._paths]

    @value.setter
    def value(self, val: list[str | Path]):
        # clear paths
        self._paths.clear()
        if val is Undefined:
            self._toggle_btn.enabled = False
            return
        self._append_paths(val)

    def _add_files_from_dialog(self):
        import os

        _paths = self.value
        if _paths:
            start_path: Path = _paths[-1]
            _start_path: str | None = os.fspath(start_path.expanduser().absolute())
        else:
            _start_path = None
        result = show_file_dialog(
            mode="rm",
            caption="Select files to add",
            filter=self._filter,
            start_path=_start_path,
            parent=self.native,
        )
        if result:
            self._append_paths(result)


class KernelEdit(Container[Container[CheckBox]]):
    """Widget for editing convolution kernel."""

    _SizeLim = 7

    def __init__(self, value=Undefined, nullable=False, **kwargs):
        widgets = list[Container[CheckBox]]()
        for _ in range(self._SizeLim):
            cnt = Container(
                widgets=[CheckBox() for _ in range(self._SizeLim)], layout="horizontal"
            )
            cnt.margins = (0, 0, 0, 0)
            widgets.append(cnt)
        self._cboxes = widgets
        self._ysize = ComboBox(value=3, choices=[3, 5, 7], name="Y size")
        self._xsize = ComboBox(value=3, choices=[3, 5, 7], name="X size")
        self._size = Container(widgets=[self._ysize, self._xsize], layout="horizontal")
        super().__init__(widgets=[self._size] + widgets, **kwargs)
        for row in self._cboxes:
            for item in row:
                item.changed.disconnect()
                item.changed.connect(self._on_value_change)
        self._size.changed.disconnect()
        self._size.changed.connect(self._set_size)
        self.value = value

    def _on_value_change(self):
        self.changed.emit(self.value)

    def _set_size(self):
        """Disable check boxes that are out of range."""
        ny = self._ysize.value
        nx = self._xsize.value
        nydrop = (self._SizeLim - ny) // 2
        nxdrop = (self._SizeLim - nx) // 2
        with self.changed.blocked():
            for ir, row in enumerate(self._cboxes):
                if ir < nydrop or self._SizeLim - nydrop <= ir:
                    for item in row:
                        item.value = False
                    row.enabled = False
                else:
                    row.enabled = True
                    for ic, item in enumerate(row):
                        if ic < nxdrop or self._SizeLim - nxdrop <= ic:
                            item.value = False
                            item.enabled = False
                        else:
                            item.enabled = True

    @property
    def value(self) -> list[list[int]]:
        ny: int = self._ysize.value
        nx: int = self._xsize.value
        arr = np.zeros((ny, nx), dtype=np.bool_)
        nydrop = (self._SizeLim - ny) // 2
        nxdrop = (self._SizeLim - nx) // 2
        for i in range(ny):
            for j in range(nx):
                arr[i, j] = self._cboxes[i + nydrop][j + nxdrop].value

        return arr.astype(np.uint8).tolist()

    @value.setter
    def value(self, val):
        if val is Undefined:
            val = np.ones((3, 3), dtype=np.bool_)
        else:
            val = np.asarray(val)
            if not set(np.unique(val)) <= {0, 1}:
                raise ValueError(
                    f"Array must contain only 0s and 1s, got {set(np.unique(val))!r}."
                )
            val = val > 0
        if val.ndim != 2 or val.dtype.kind not in "uifb":
            raise ValueError("Array must be 2D and numeric.")
        ny, nx = val.shape
        lim = self._SizeLim
        if ny > lim or nx > lim:
            raise ValueError(f"Array must be {lim}x{lim} or smaller.")
        elif ny % 2 == 0 or nx % 2 == 0:
            raise ValueError(f"Array must be odd in both dimensions, got {(ny, nx)}.")
        yoffset, xoffset = (lim - ny) // 2, (lim - nx) // 2
        with self.changed.blocked():
            self._ysize.value, self._xsize.value = ny, nx
            self._set_size()
            for y in range(ny):
                for x in range(nx):
                    self._cboxes[y + yoffset][x + xoffset].value = bool(val[y, x])
        self.changed.emit(val.tolist())


@contextmanager
def _signals_blocked(obj: QtW.QWidget):
    before = obj.blockSignals(True)
    try:
        yield
    finally:
        obj.blockSignals(before)


def _not(x: Qt.CheckState) -> Qt.CheckState:
    if x == Qt.CheckState.Checked:
        state = Qt.CheckState.Unchecked
    else:
        state = Qt.CheckState.Checked
    return state


class _ListWidget(QtW.QListWidget):
    def __init__(self, parent: QtW.QWidget | None = None):
        super().__init__(parent)
        self.setSelectionMode(QtW.QAbstractItemView.SelectionMode.NoSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.itemEntered.connect(self._item_entered)
        self._mouse_start: int | None = None
        self._dragged = False
        self._old_state = None

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() & Qt.MouseButton.LeftButton:
            self._dragged = False
            self._mouse_start = self.indexAt(e.pos()).row()
            self._old_state = [self.item(i).checkState() for i in range(self.count())]
        return super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        if not self._dragged and self._mouse_start is not None:
            i = self._mouse_start
            if item := self.item(i):
                item.setCheckState(_not(self._old_state[i]))
            self._mouse_start = None
            return
        self._mouse_start = None
        return super().mouseReleaseEvent(e)

    def _item_entered(self, item: QtW.QListWidgetItem):
        if self._mouse_start is None:
            return None
        self._dragged = True
        r0 = self._mouse_start
        r1 = self.indexFromItem(item).row()
        r0, r1 = sorted([r0, r1])
        for i in range(r0):
            self._set_check_state(i, self._old_state[i])
        for i in range(r0, r1 + 1):
            self._set_check_state(i, _not(self._old_state[i]))
        for i in range(r1 + 1, self.count()):
            self._set_check_state(i, self._old_state[i])

    def _set_check_state(self, i: int, state: Qt.CheckState):
        """Safely set check state of the item at the given index."""
        if item := self.item(i):
            item.setCheckState(state)

    def sizeHint(self) -> QtCore.QSize:
        size = super().sizeHint()
        size.setHeight(int(size.height() / 2))
        return size

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if (
            e.key() == Qt.Key.Key_A
            and e.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            # check all
            for i in range(self.count()):
                self.item(i).setCheckState(Qt.CheckState.Checked)
            return None
        return super().keyPressEvent(e)


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
        item.setCheckState(_not(item.checkState()))

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
            nullable=False,
            **base_widget_kwargs,
        )

    @property
    def value(self) -> Any:
        return super().value

    @value.setter
    def value(self, value) -> None:
        if value is None:
            value = ()
        if isinstance(value, (list, tuple)) and self._allow_multiple:
            if any(v not in self.choices for v in value):
                raise ValueError(
                    f"{value!r} is not a valid choice. must be in {self.choices}"
                )
        elif value not in self.choices:
            raise ValueError(
                f"{value!r} is not a valid choice. must be in {self.choices}"
            )
        return ValueWidget.value.fset(self, value)  # type: ignore
