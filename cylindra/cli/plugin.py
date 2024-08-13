from __future__ import annotations

from pathlib import Path

from cylindra.cli._base import ParserBase


class ParserPlugin(ParserBase):
    """
    cylindra plugin ...

    cylindra plugin list [u bold green]filter[/u bold green]
        List all plugins. If [u bold green]filter[/u bold green] is provided, only list
        plugins that contain the filter.
    cylindra plugin new [u bold green]path[/u bold green]
        Create a new plugin at the given [u bold green]path[/u bold green] (current
        directory by default).
    """

    def __init__(self):
        super().__init__(
            prog="cylindra plugin",
            description="Create or list plugins.",
        )
        self.add_argument("operation", nargs="?", type=str)
        self.add_argument("param", nargs="?", type=str)

    def run_action(
        self,
        operation: str | None = None,
        param: str | None = None,
        **kwargs,
    ):
        match operation:
            case "new":
                self.create_plugin(param)
            case "list":
                self.list_plugins(param)
            case _:
                self.print_help()

    def create_plugin(self, param: str | None = None):
        from cookiecutter.main import cookiecutter

        if param is None:
            path = Path.cwd()
        else:
            path = Path(param)
        template_path = str(Path(__file__).parent / "_cookiecutter_template")
        output_dir = str(path)
        cookiecutter(template_path, output_dir=output_dir)

    def list_plugins(self, param: str | None = None):
        import rich

        from cylindra.plugin._find import iter_plugin_info

        names = list[str]()
        versions = list[str]()
        for info in iter_plugin_info():
            if param and param not in info.value:
                continue
            names.append(
                f"[bold green]{info.value}[/bold green] [gray]({info.name})[/gray]"
            )
            versions.append(f"[yellow]{info.version}[/yellow]")
        name_length_max = max(len(name) for name in names)
        lines = [
            f"{name:<{name_length_max}} {version}"
            for name, version in zip(names, versions, strict=False)
        ]
        rich.print("\n".join(lines))
