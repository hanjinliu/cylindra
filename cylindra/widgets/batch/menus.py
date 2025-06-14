from magicclass import MagicTemplate, abstractapi, magicmenu
from magicgui.types import Separator


@magicmenu
class BatchSubtomogramAnalysis(MagicTemplate):
    """Subtomogram analysis over multiple projects."""

    average_all = abstractapi()
    average_groups = abstractapi()
    split_and_average = abstractapi()
    calculate_fsc = abstractapi()
    sep0 = Separator
    classify_pca = abstractapi()


@magicmenu
class BatchRefinement(MagicTemplate):
    """Refinement of subtomograms."""

    align_all = abstractapi()
    align_all_template_free = abstractapi()


@magicmenu
class BatchLoaderMenu(MagicTemplate):
    """Handle batch loader"""

    split_loader = abstractapi()
    filter_loader = abstractapi()


@magicmenu(name="Macro")
class Macro(MagicTemplate):
    """Show macro of batch analyzer."""

    show_macro = abstractapi()
    show_native_macro = abstractapi()
