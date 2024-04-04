from magicclass import (
    abstractapi,
    bind_key,
    do_not_record,
    field,
    magicmenu,
    magictoolbar,
    set_design,
    vfield,
)
from magicclass.logging import getLogger
from magicclass.widgets import Separator

from cylindra.components.picker import AutoCorrelationPicker
from cylindra.widget_utils import change_viewer_focus
from cylindra.widgets.subwidgets._child_widget import ChildWidget

_Logger = getLogger("cylindra")


@magictoolbar(labels=False)
class CylindraToolbar(ChildWidget):
    """Frequently used operations."""

    register_path = abstractapi()

    @set_design(icon="ant-design:fast-forward-filled")
    @bind_key("F2")
    @do_not_record
    def open_runner(self):
        """Run cylindrical fitting algorithm with various settings."""
        return self._get_main()._runner.show(run=False)

    sep0 = field(Separator)

    @set_design(icon="fe:target")
    @bind_key("F3")
    @do_not_record
    def pick_next(self):
        """Automatically pick cylinder center using previous two points."""
        picker = self.Adjust._get_picker()
        main = self._get_main()
        points = main._reserved_layers.work.data
        if len(points) < 2:
            raise IndexError("Auto picking needs at least two points.")
        imgb = max(main.tomogram.multiscaled, key=lambda x: x[0])[1]
        scale = imgb.scale.x
        next_point = picker.iter_pick(imgb, points[-1], points[-2]).next()
        main._reserved_layers.work.add(next_point)
        change_viewer_focus(main.parent_viewer, next_point / scale, scale)
        return None

    @magicmenu(icon="carbon:settings-adjust", record=False)
    class Adjust(ChildWidget):
        """
        Adjust auto picker parameters.

        Attributes
        ----------
        interval : nm
            Interval (nm) of auto picking.
        max_angle : float
            Maximum angle (degree) to search in auto picking.
        angle_step : float
            Step of angle (degree) to search in auto picking.
        max_shifts : nm
            Maximum shift (nm) to search in auto picking.
        """

        interval = vfield(80.0, widget_type="FloatSlider").with_options(min=10, max=200)  # fmt: skip
        max_angle = vfield(12.0, widget_type="FloatSlider").with_options(min=1.0, max=40.0, step=0.5)  # fmt: skip
        angle_step = vfield(1.0, widget_type="FloatSlider").with_options(min=0.5, max=5.0, step=0.1)  # fmt: skip
        max_shifts = vfield(20.0).with_options(min=1.0, max=50.0, step=0.5)

        def _get_picker(self) -> AutoCorrelationPicker:
            """Make a picker with current parameters."""
            cfg = self._get_main().default_config
            return AutoCorrelationPicker(self.interval, self.max_angle, self.angle_step, self.max_shifts, cfg)  # fmt: skip

    sep1 = field(Separator)

    clear_current = abstractapi()
    clear_all = abstractapi()

    sep2 = field(Separator)

    @do_not_record
    @set_design(icon="gg:undo")
    def undo(self):
        """Undo last action."""
        if len(self.macro.undo_stack["undo"]) == 0:
            raise RuntimeError("Undo stack is empty.")
        expr = self.macro[-1]
        self.macro.undo()
        return _Logger.print_html(f"Undo: <code>{expr}</code>")

    @do_not_record
    @set_design(icon="gg:redo")
    def redo(self):
        """Redo last undo action."""
        if len(self.macro.undo_stack["redo"]) == 0:
            raise RuntimeError("Redo stack is empty.")
        self.macro.redo()
        expr = self.macro[-1]
        return _Logger.print_html(f"Redo: <code>{expr}</code>")
