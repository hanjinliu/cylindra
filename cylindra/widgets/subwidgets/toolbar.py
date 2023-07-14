from pathlib import Path
from magicclass import (
    do_not_record,
    magicmenu,
    magictoolbar,
    field,
    vfield,
    MagicTemplate,
    set_design,
    bind_key,
    abstractapi,
)
from magicclass.widgets import Separator
from magicclass.logging import getLogger
from cylindra.components.picker import AutoCorrelationPicker

ICON_DIR = Path(__file__).parent.parent / "icons"

_Logger = getLogger("cylindra")


@magictoolbar(labels=False)
class CylindraToolbar(MagicTemplate):
    """Frequently used operations."""

    register_path = abstractapi()
    open_runner = abstractapi()
    sep0 = field(Separator)
    pick_next = abstractapi()

    @magicmenu(icon=ICON_DIR / "adjust_intervals.svg", record=False)
    class Adjust(MagicTemplate):
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

        interval = vfield(80.0, widget_type="FloatSlider").with_options(min=10, max=1000)  # fmt: skip
        max_angle = vfield(12.0, widget_type="FloatSlider").with_options(min=1.0, max=40.0, step=0.5)  # fmt: skip
        angle_step = vfield(1.0, widget_type="FloatSlider").with_options(min=0.5, max=5.0, step=0.1)  # fmt: skip
        max_shifts = vfield(20.0).with_options(min=1.0, max=50.0, step=0.5)

        def _get_picker(self) -> AutoCorrelationPicker:
            """Make a picker with current parameters."""
            return AutoCorrelationPicker(self.interval, self.max_angle, self.angle_step, self.max_shifts)  # fmt: skip

    sep1 = field(Separator)

    clear_current = abstractapi()
    clear_all = abstractapi()

    sep2 = field(Separator)

    @do_not_record
    @set_design(icon=ICON_DIR / "undo.svg")
    @bind_key("Ctrl+Z")
    def undo(self):
        """Undo last action."""
        if len(self.macro.undo_stack["undo"]) == 0:
            raise RuntimeError("Undo stack is empty.")
        expr = self.macro[-1]
        self.macro.undo()
        return _Logger.print_html(f"Undo: <code>{expr}</code>")

    @do_not_record
    @set_design(icon=ICON_DIR / "redo.svg")
    @bind_key("Ctrl+Y")
    def redo(self):
        """Redo last undo action."""
        if len(self.macro.undo_stack["redo"]) == 0:
            raise RuntimeError("Redo stack is empty.")
        self.macro.redo()
        expr = self.macro[-1]
        return _Logger.print_html(f"Redo: <code>{expr}</code>")
