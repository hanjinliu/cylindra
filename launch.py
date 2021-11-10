from mtprops import start
import napari
ui = start()
ui.parent_viewer.update_console({"ui": ui})
napari.run()