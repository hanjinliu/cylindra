import sys
import numpy as np
import impy as ip
import argparse
from mtprops import start

def get_args():
    parser = argparse.ArgumentParser(description="MTProps")
    from mtprops.const import GVar

    for name, type_ in GVar.__annotations__.items():
        parser.add_argument("--" + name, type=type_)
    
    return parser.parse_args()
    
    
if __name__ == "__main__":
    ui = start()
    ui.parent_viewer.update_console({"ui": ui, "ip": ip, "np": np})
    if len(sys.argv) > 1:
        args = get_args()
        ui["Global_variables"].changed()
        mgui = ui["Global_variables"].mgui
        for key, value in args.__dict__.items():
            if value is None:
                continue
            widget = getattr(mgui, key)
            widget.value = value
        mgui[-1].changed()
    ui.parent_viewer.show(block=True)