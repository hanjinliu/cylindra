from magicclass import magicmenu, MagicTemplate, abstractapi

@magicmenu
class BatchSubtomogramAnalysis(MagicTemplate):
    average_all = abstractapi()
    calculate_fsc = abstractapi()

@magicmenu
class BatchRefinement(MagicTemplate):
    """Refinement of subtomograms."""
    align_all = abstractapi()

@magicmenu(name="Macro")
class Macro(MagicTemplate):
    """Macro operations."""
    show_macro = abstractapi()
    show_native_macro = abstractapi()
