import sys
import numpy as np
import impy as ip
import argparse
from mtprops import start

def get_args():
    parser = argparse.ArgumentParser(description="MTProps")
    parser.add_argument("--project", type=str, default="None")
    parser.add_argument("--globals", type=str, default="None")
    return parser.parse_args()
    
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = get_args()
        project_file = None if args.project == "None" else args.project
        globals_file = None if args.globals == "None" else args.globals
        ui = start(project_file=project_file, globals_file=globals_file)
    else:
        ui = start()

    ui.parent_viewer.update_console({"ui": ui, "ip": ip, "np": np})
    ui.parent_viewer.show(block=True)