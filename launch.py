from mtprops import start
import napari
import sys

if __name__ == "__main__":
    ui = start()
    sys.exit(napari.run())