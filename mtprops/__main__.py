import argparse

class Namespace(argparse.Namespace):
    project: str
    globals: str

class Args(argparse.ArgumentParser):
    def __init__(self):
        from ._info import __version__
        super().__init__(description="Command line interface of MTProps.")
        self.add_argument("--project", type=str, default="None")
        self.add_argument("--globals", type=str, default="None")    
        self.add_argument("-v", "--version", action="version", version=f"MTProps version {__version__}")
    
    @classmethod
    def from_args(cls) -> Namespace:
        return cls().parse_args()

def main():
    args = Args.from_args()
    
    project_file = None if args.project == "None" else args.project
    globals_file = None if args.globals == "None" else args.globals
    
    from . import start
    ui = start(project_file=project_file, globals_file=globals_file)

    import numpy as np
    import impy as ip
    ui.parent_viewer.update_console({"ui": ui, "ip": ip, "np": np})
    ui.parent_viewer.show(block=True)

if __name__ == "__main__":
    main()
