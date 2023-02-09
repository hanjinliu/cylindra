from typing import Iterator, Union, TYPE_CHECKING, Annotated
from magicclass import (
    magicclass, magicmenu, field, MagicTemplate, abstractapi
)
from magicclass.widgets import Separator

@magicmenu(name="File")
class File(MagicTemplate):
    add_children = abstractapi()
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
