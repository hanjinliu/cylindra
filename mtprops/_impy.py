try:
    import impy
    from impy import lazy_imread
except ImportError as e:
    raise ImportError(f"impy is not correctly imported, or in a wrong path. "
                      f"Original ImportError: {e}")
