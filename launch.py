import impy as ip
from mtprops import start

if __name__ == "__main__":
    ui = start(ip.gui.viewer)
    ui.parent_viewer.update_console({"ui": ui, "ip": ip})
    ip.gui.viewer.show(block=True)