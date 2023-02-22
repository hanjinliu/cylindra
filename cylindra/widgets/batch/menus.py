from magicclass import magicmenu, field, MagicTemplate, abstractapi
from magicclass.widgets import Separator

@magicmenu(name="File")
class Projects(MagicTemplate):
    open_constructor = abstractapi()
    sep0 = field(Separator)
    load_project = abstractapi()
    save_project = abstractapi()

@magicmenu
class Splines(MagicTemplate):
    """Analyze splines collectively."""
    view_localprops = abstractapi()

@magicmenu(name="Subtomogram Analysis")
class SubtomogramAnalysis(MagicTemplate):
    """Analysis of subtomograms."""
    average_all = abstractapi()
    sep0 = field(Separator)
    align_all = abstractapi()
    calculate_fsc = abstractapi()

@magicmenu(name="Macro")
class Macro(MagicTemplate):
    """Macro operations."""
    show_macro = abstractapi()
    show_native_macro = abstractapi()
