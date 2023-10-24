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


def test_check_boxes():
    widget = _widget_ext.CheckBoxes(choices=["a", "b", "c"])
    widget.value = ["a", "b"]
    widget.value = ["b", "a"]


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
