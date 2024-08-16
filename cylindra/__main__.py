from __future__ import annotations

import sys

from cylindra import cli


def main(viewer=None, ignore_sys_exit: bool = False):
    argv = sys.argv[1:]
    cli.set_current_viewer(viewer)
    try:
        cli.exec(argv)
    except SystemExit as e:
        if ignore_sys_exit:
            return
        else:
            raise e


if __name__ == "__main__":
    main()
    sys.exit()
