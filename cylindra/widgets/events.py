from psygnal import Signal, SignalGroup


class MainWidgetEvents(SignalGroup):
    """Events for CylindraMainWidget."""

    tomogram_initialized = Signal()
    project_loaded = Signal()
    cleared = Signal()
