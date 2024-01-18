from __future__ import annotations

import sys

from cylindra import cli


def main(viewer=None, ignore_sys_exit: bool = False):
    argv = sys.argv[1:]
    cli.set_current_viewer(viewer)
    try:
        match argv:
            case "open", *args:
                cli.ParserOpen().parse(args)
            case "preview", *args:
                cli.ParserPreview().parse(args)
            case "run", *args:
                cli.ParserRun().parse(args)
            case "average", *args:
                cli.ParserAverage().parse(args)
            case "new", *args:
                cli.ParserNew().parse(args)
            case "config", *args:
                cli.ParserConfig().parse(args)
            case "find", *args:
                cli.ParserFind().parse(args)
            case "workflow", *args:
                cli.ParserWorkflow().parse(args)
            case args:
                cli.ParserNone().parse(args)
    except SystemExit as e:
        if ignore_sys_exit:
            return
        else:
            raise e


if __name__ == "__main__":
    main()
    sys.exit()
