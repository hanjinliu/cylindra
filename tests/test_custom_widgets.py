import pytest
from pytestqt.qtbot import QtBot

from cylindra.widgets import _widget_ext


def test_multi_file_edit():
    widget = _widget_ext.MultiFileEdit()
    widget._append_paths(["C:/test1.tif", "C:/test2.tif"])
    assert len(widget.value) == 2
    widget._toggle_delete_checkboxes()
    widget._toggle_delete_checkboxes()
    widget._toggle_delete_checkboxes()
    widget._paths[0][0].value = True
    widget._delete_checked_paths()
    assert len(widget.value) == 1


def test_construction():
    _widget_ext.SingleRotationEdit()


def test_check_boxes(qtbot: QtBot):
    from qtpy import QtCore

    widget = _widget_ext.CheckBoxes(choices=["a", "b", "c"])
    widget.native.show()
    qtbot.addWidget(widget.native)

    assert widget.choices == ("a", "b", "c")
    widget.value = ["a", "b"]
    assert widget.value == ["a", "b"]
    widget.value = ["b", "a"]
    assert widget.value == ["a", "b"]
    with pytest.raises(ValueError):
        widget.value = ["a", "d"]
    widget.choices = ["x", "y"]
    qtbot.keyClick(widget.native, QtCore.Qt.Key.Key_Down)
    qtbot.keyClick(widget.native, QtCore.Qt.Key.Key_Up)
    qtbot.keyClick(
        widget.native, QtCore.Qt.Key.Key_A, QtCore.Qt.KeyboardModifier.ControlModifier
    )
    pos0 = widget.native.rect().topLeft() + QtCore.QPoint(3, 3)
    pos1 = widget.native.rect().topLeft() + QtCore.QPoint(3, 23)
    qtbot.mousePress(widget.native, QtCore.Qt.MouseButton.LeftButton, pos=pos0)
    # move by 20 pixels down
    qtbot.mouseMove(widget.native, pos=pos1)
    qtbot.mouseRelease(widget.native, QtCore.Qt.MouseButton.LeftButton)


def test_ctf_params(qtbot: QtBot):
    widget = _widget_ext.CTFParams()
    qtbot.addWidget(widget.native)
    widget.value = {
        "kv": 200,
        "spherical_aberration": 2.0,
        "defocus": -3.2,
        "bfactor": 0.1,
        "correct": "none",
    }
    assert widget.value == pytest.approx(
        {
            "kv": 200,
            "spherical_aberration": 2.0,
            "defocus": -3.2,
            "bfactor": 0.1,
            "correct": "none",
        }
    )
    widget._has_input.value = False
    widget._has_input.value = True


def test_random_seed_edit():
    widget = _widget_ext.RandomSeedEdit()
    widget._btn.clicked()


def test_kernel_edit():
    widget = _widget_ext.KernelEdit()
    for value in [
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[0, 0, 0], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
    ]:
        widget.value = value
        assert widget.value == value


def test_index_edit():
    widget = _widget_ext.IndexEdit()
    for value in [
        "1, 2, npf",
        "1, npf,",
        "1, slice(10, None), N",
        "[1, slice(None, 4)]",
        "slice(4)",
        "10",
    ]:
        widget.value = value
        widget.value  # noqa: B018

    for value in ["slice(1.1, 5), 4", "3, 5, 3.4", "5/2, 10"]:
        widget.value = value
        with pytest.raises(ValueError):
            widget.value  # noqa: B018
