from __future__ import annotations

from pathlib import Path

from cylindra.cli._base import ParserBase


class ParserPrepDefaults(ParserBase):
    """cylindra prep-defaults ...

    cylindra prep-defaults [u bold green]path[/u bold green]
        Create a '.cylindra-defaults.toml' file under directory [u bold green]path[/u bold green].

    [u bold cyan]options[/u bold cyan]
        [bold]--validate, -v[/bold]
            Validate the existing '.cylindra-defaults.toml' file instead of creating a new one.
    """

    def __init__(self):
        super().__init__(
            prog="cylindra prep-defaults",
            description="Create a '.cylindra-defaults.toml' file.",
        )
        self.add_argument("path", nargs="?", default=".", type=str)
        self.add_argument(
            "--validate", "-v", action="store_true", help="Validate the created file."
        )

    def run_action(
        self,
        path: str = ".",
        validate: bool = False,
        **kwargs,
    ):
        from cylindra import __version__

        dirpath = Path(path).expanduser().resolve()
        toml_path = dirpath / ".cylindra-defaults.toml"
        if validate:
            from cylindra.project import TomogramDefaults

            toml_defaults = TomogramDefaults.from_dir(dirpath, parent_ok=False)
            # NOTE: if the toml format is invalid, an exception will be raised here.
            if toml_defaults is None:
                print(f"No '.cylindra-defaults.toml' found under {dirpath}")
            else:
                print("'.cylindra-defaults.toml' is valid.")
        else:
            txt = TOML_TEMPLATE.replace("$(version)", __version__)
            toml_path.write_text(txt, encoding="utf-8")


TOML_TEMPLATE = """# Created by cylindra=v$(version)
# Default settings for tomograms in this directory and the sub-directories. These
# settings will be read when opening tomograms. All the entries are optional; you can
# delete the ones you don't need.

# if given, override the image pixel size (nm/pixel)
scale = 0.54

# whether to invert the tomogram contrast to make dark background
invert = true

# if given, path to the reference image file (such as binned tomogram or denoised
# tomomgram). Can use format string to include tomogram file stem, and wildcards to
# match files. For example, for tomogram "TOMO.mrc", "{}_*.mrc" will match files such as
# "TOMO_denoised.mrc" or "TOMO_bin4.mrc".
image_reference = "{}_*.mrc"

# whether to invert the reference image contrast to make dark background
invert_reference = true

# bin sizes to make multiscale tomograms.
bin_size = [2, 4]

# filter to apply to the reference image or binned tomograms.
# Options are: "Lowpass", "Gaussian", "DoG" or "LoG".
filter = "Lowpass"

# missing wedge specification
[missing_wedge]
kind = "y"
range = [-57, 57]
"""
