from cylindra.plugin import register_function
from cylindra.widgets import CylindraMainWidget


@register_function
def my_plugin_function(ui: CylindraMainWidget):
    ui.logger.print("Hello world!")
