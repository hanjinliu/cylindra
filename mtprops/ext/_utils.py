from __future__ import annotations
import subprocess
import shutil
from typing import Callable

class CommandNotFound(RuntimeError):
    """Raised if command is not found in the system paths."""


def assert_command_exists(cmd: str):
    if shutil.which(cmd) is None:
        raise CommandNotFound(f"Command {cmd} was not found.")


def translate_command(cmd: str) -> Callable[[str], Callable[..., None]]:
    """
    Convert command into Python function.
    If command is ``cmd input -op option``, it corresponds to ``cmd(input, op=option)``.
    """
    def _run(*args, **kwargs):
        assert_command_exists(cmd)
        options = sum(([f"-{k}", str(v)] for k, v in kwargs.items()), start=[])
        subprocess.run([cmd] + list(args) + options)
        return None
    _run.__name__ = cmd
    return _run