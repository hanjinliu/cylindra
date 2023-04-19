from magicclass import magicmenu, MagicTemplate, abstractapi, field
from magicclass.widgets import Separator


@magicmenu
class BatchSubtomogramAnalysis(MagicTemplate):
    average_all = abstractapi()
    calculate_fsc = abstractapi()
    sep0 = field(Separator)
    classify_pca = abstractapi()


@magicmenu
class BatchRefinement(MagicTemplate):
    """Refinement of subtomograms."""

    align_all = abstractapi()


@magicmenu
class BatchLoaderMenu(MagicTemplate):
    """Handle batch loader"""

    split_loader = abstractapi()
    filter_loader = abstractapi()


@magicmenu(name="Macro")
class Macro(MagicTemplate):
    """Macro operations."""

    show_macro = abstractapi()
    show_native_macro = abstractapi()
