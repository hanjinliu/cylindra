try:
    import impy
    from impy import lazy_imread
except ImportError as e:
    raise ImportError(f"impy is not correctly imported, or in a wrong path. "
                      f"Original ImportError: {e}")

try:
    import magicclass as mcls
    from magicclass import magicclass, field, button_design, click, set_options
    from magicclass.widgets import Figure, TupleEdit, CheckButton
except ImportError as e:
    raise ImportError(f"magicclass is not correctly imported, or in a wrong path. "
                      f"Original ImportError: {e}")