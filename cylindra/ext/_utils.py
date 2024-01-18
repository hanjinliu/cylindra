from __future__ import annotations

import shutil
import subprocess
from typing import Callable


class CommandNotFound(RuntimeError):
    """Raised if command is not found in the system paths."""


class CommandExecutionError(RuntimeError):
    """Raised if command ended with error."""


class CLICommand(Callable):
    def __init__(self, cmd: str):
        self._cmd = cmd

    def __call__(self, *args, **kwargs):
        if not self.available():
            raise CommandNotFound(f"Command {self._cmd} was not found.")
        options = []
        for k, v in kwargs.items():
            options.append(f"-{k}")
            if isinstance(v, bool):
                continue
            options.append(str(v))
        process = subprocess.run(
            [self._cmd] + list(args) + options, capture_output=True, check=False
        )
        out = process.stdout.decode()
        err = process.stderr.decode()
        if err:
            raise CommandExecutionError(err)
        elif out.startswith("ERROR"):
            raise CommandExecutionError(out)
        return out

    def available(self) -> bool:
        return shutil.which(self._cmd) is not None


def translate_command(cmd: str) -> CLICommand:
    """
    Convert command into Python function.
    If command is ``cmd input -op option``, it corresponds to ``cmd(input, op=option)``.
    """

    return CLICommand(cmd)
