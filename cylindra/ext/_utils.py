from __future__ import annotations
import subprocess
import shutil
from typing import Callable

class CommandNotFound(RuntimeError):
    """Raised if command is not found in the system paths."""

class CommandExecutionError(RuntimeError):
    """Raised if command ended with error."""


def assert_command_exists(cmd: str):
    """Raise CommandNotFound exception if command ``cmd`` was not found."""
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
        process = subprocess.run([cmd] + list(args) + options, capture_output=True)
        out = process.stdout.decode()
        err = process.stderr.decode()
        if err:
            raise CommandExecutionError(err)
        elif out.startswith("ERROR"):
            raise CommandExecutionError(out)
        return out
    _run.__name__ = cmd
    return _run