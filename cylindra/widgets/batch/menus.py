from magicclass import MagicTemplate, abstractapi, field, magicmenu
from magicclass.widgets import Separator


@magicmenu
class BatchSubtomogramAnalysis(MagicTemplate):
    """Subtomogram analysis over multiple projects."""

    average_all = abstractapi()
    average_groups = abstractapi()
    split_and_average = abstractapi()
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
    """Show macro of batch analyzer."""

    show_macro = abstractapi()
    show_native_macro = abstractapi()
